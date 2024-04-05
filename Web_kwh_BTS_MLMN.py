import pandas as pd
import streamlit as st
import json
import requests
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# Streamlit UI
st.title("TOOL xem thông tin điện trạm BTS ký HĐ trực tiếp với EVN miền Nam")

global password
password=st.secrets["pass_EVNSPC"]

# Function to view info the electricity in daily
def day_view_kwh():
    global password
    if not MaKH or MaKH.strip() == "":
        st.warning("Vui lòng nhập đầy đủ thông tin Mã khách hàng (PB hoặc PK, 13 chữ) bên dưới")
        return
    
    st.empty()
    st.markdown(f"**Thông tin điện tiêu thụ theo ngày của trạm BTS có MaKH {MaKH}, phản hồi từ web: https://cskh.evnspc.vn**")

    # Create JSON data
    json_data = {
        "MaDonViKH": "MOBIPHONEMLMN",
        "MaKhachHang": MaKH,
        "TenDangNhap": MaKH,
        "MatKhau": password,
        "MaChucNang": "DOGHIXA",
        "TuNgayChiSoChot": From_date.strftime("%d-%m-%Y"),
        "DenNgayChiSoChot": To_date.strftime("%d-%m-%Y")
    }

    # Set up headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": "api.cskh.evnspc.vn",
        "User-Agent": "python-requests/2.26.0"
    }

    # Set up the URL
    url = "https://www.cskh.evnspc.vn/TraCuu/KhachHangTrongDiem"

    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(json_data))
    result_df = pd.DataFrame()
    # Check if the response is successful (status code 200)
    if response.status_code == 200:
    # Parse the response as JSON
        response_data = response.json() 
        # Tính toán số điện tiêu thụ mỗi ngày dựa trên PGIAOTONG của ngày đó trừ PGIAOTONG của ngày trước đó
        # Check if response_data is a list and has more than one element
        if isinstance(response_data, list) and len(response_data) > 1:
        # Loop through each JSON entry
            valid_electricity_consumption = []
            valid_dates = []
            for i in range(1, len(response_data)):
                # Check if 'PGIAOTONG' key exists in each entry
                if 'PGIAOTONG' in response_data[i] and 'PGIAOTONG' in response_data[i-1]:
                    consumption = response_data[i]['PGIAOTONG'] - response_data[i-1]['PGIAOTONG']
                    current_iteration_data = {
                        'MaKH': MaKH,
                        'Ngay': response_data[i]['TIME'],
                        'kwh': consumption
                    }
                    # Convert current_iteration_data to DataFrame
                    current_iteration_df = pd.DataFrame([current_iteration_data])
                    # Append data to the result DataFrame
                    result_df = pd.concat([result_df, current_iteration_df], axis=0, ignore_index=True) 
                    if consumption >= 0:
                        valid_electricity_consumption.append(round(consumption, 2))  # Làm tròn số kWh
                        valid_dates.append(response_data[i]['TIME'])

            # Số lượng ngày có kWh hợp lệ
            num_valid_days = len(valid_dates)

            st.markdown("**Bảng kết quả như sau:**")
            st.dataframe(result_df,hide_index=True)
                        
            # Biến đổi dữ liệu thành numpy array
            X = np.array(valid_electricity_consumption).reshape(-1, 1)

            # Sử dụng Isolation Forest để phát hiện ngoại lệ
            clf = IsolationForest(contamination=0.02)  # 5% là ngoại lệ
            clf.fit(X)

            # Dự đoán xác suất là ngoại lệ cho mỗi điểm dữ liệu
            outliers = clf.predict(X)

            # Hiển thị biểu đồ chỉ khi có dữ liệu hợp lệ
            if num_valid_days > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(valid_dates, valid_electricity_consumption, marker='o', linestyle='-')
                ax.set_title(f'Biểu đồ số kwh theo ngày của MaKH {MaKH}')
                ax.set_xlabel('Ngày')
                ax.set_ylabel('Số kwh')
                ax.tick_params(axis='x', rotation=90)  # Xoay nhãn trục x để dễ đọc hơn
                # Đánh dấu các điểm ngoại lệ trên biểu đồ
                for i in range(len(outliers)):
                    if outliers[i] == -1:  # Ngoại lệ
                        ax.scatter(valid_dates[i], valid_electricity_consumption[i], color='red')
                        ax.text(valid_dates[i], valid_electricity_consumption[i], 'Bất thường', fontsize=10, verticalalignment='bottom')

                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.write("Không có dữ liệu hợp lệ để vẽ biểu đồ.")
        else:
            if 'content' in response_data:
                st.error(f"Kết quả: lỗi. {response_data['content']}")
            else:
                st.error(f"Kết quả: dữ liệu trả về là [] (null)")
    else:
        st.error(f"Lỗi khi thực hiện request: {response.status_code}")

# Function to view info the electricity in daily
def day_get_kwh():
    global password
    # Record the start time
    start_time = time.time()
    # Đọc dữ liệu từ tệp Excel
    get_file_path = uploaded_file
    if get_file_path is None:
        st.sidebar.warning("Vui lòng tải file dữ liệu MaKH cần lấy thông tin kwh trước.")
        return
    
    st.sidebar.empty()
    st.sidebar.success("Đang thực hiện lấy thông tin điện ...")

    data = pd.read_excel(get_file_path)
    num_unique_MaKH = len(data[data["MaKH"].notnull()]["MaKH"].unique())
    num_rows = data.shape[0]
    num_rows_null=num_rows-num_unique_MaKH
    data = data.dropna(subset=['MaKH'])
    # Remove duplicates based on the specified column
    data = data.drop_duplicates(subset=['MaKH'])

    # Hiển thị thông tin DataFrame của file tải lên
    st.empty()
    st.markdown("**Thông tin của file MaKH tải lên:**")
    st.write(f"Có tổng cộng {num_rows} hàng. Trong đó:")
    st.write("-Số MaKH có dữ liệu hợp lệ là:",num_unique_MaKH)
    st.write("-Số MaKH có dữ liệu không hợp lệ (null-để trống) là:",num_rows_null)

    # Common JSON data
    json_data_common = {
        "MaDonViKH": "MOBIPHONEMLMN",
        "MatKhau": password,
        "MaChucNang": "DOGHIXA",
        "TuNgayChiSoChot": From_date.strftime("%d-%m-%Y"),
        "DenNgayChiSoChot": To_date.strftime("%d-%m-%Y")
    }

    # Set up common headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": "api.cskh.evnspc.vn",
        "User-Agent": "python-requests/2.26.0"
    }

    # Set up the URL
    url = "https://www.cskh.evnspc.vn/TraCuu/KhachHangTrongDiem"

    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame()

    # Iterate over "MaKH" values in the DataFrame
    # Iterate over "MaKH" values in the DataFrame
    for index, row in data.iterrows():
        # Create JSON data for the specific combination
        json_data = {
            **json_data_common,
            "MaKhachHang": row['MaKH'],
            "TenDangNhap": row['MaKH']
        }

        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(json_data))
        
        # Check if the response status is OK (200)
        if response.status_code == 200:
            # Parse the response as JSON
            response_data = response.json()
            
            # Check if response_data is a list and has more than one element
            if isinstance(response_data, list) and len(response_data) > 1:
                # Loop through each JSON entry
                for i in range(1, len(response_data)):
                    # Check if 'PGIAOTONG' key exists in each entry
                    if 'PGIAOTONG' in response_data[i] and 'PGIAOTONG' in response_data[i-1]:
                        # Calculate electricity consumption for each day
                        consumption = response_data[i]['PGIAOTONG'] - response_data[i-1]['PGIAOTONG']
                        current_iteration_data = {
                            'MaKH': row['MaKH'],
                            'Ngay': response_data[i]['TIME'],
                            'kwh': consumption,
                        }
                        # Convert current_iteration_data to DataFrame
                        current_iteration_df = pd.DataFrame([current_iteration_data])
                        # Append data to the result DataFrame
                        result_df = pd.concat([result_df, current_iteration_df], axis=0, ignore_index=True) 
                    else:
                        st.write(f"Không tìm thấy khóa 'PGIAOTONG' trong dữ liệu phản hồi cho MaKhachHang {row['MaKH']}")
            
        else:
            st.write(f"Yêu cầu thất bại cho MaKhachHang {row['MaKH']}")
            
    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Display the elapsed time
    st.markdown(f"**Thời gian thực hiện lấy thông tin: {elapsed_time:.2f} giây**")

    if result_df is not None:
        # Get the current date and time in a specific format
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Đường dẫn lưu trữ file mới trong thư mục Downloads trên ổ C        
        #month_str = "_".join(map(str, Month))
        save_path = os.path.join(os.path.expanduser("~"), "Downloads", f"{current_datetime}_result_get_kwh.xlsx")
        #save_path = os.path.join(os.path.expanduser("~"), "Downloads", f"result_get_kwh_{Year}_{month_str}.xlsx")
        #save_path = os.path.join(os.path.expanduser("~"), "Downloads", "result_get_kwh_{}_{}.xlsx".format(Year, month_str))

        # Lưu kết quả vào đường dẫn mới
        result_df.to_excel(save_path, index=False)
        st.sidebar.empty()
        st.sidebar.success(f"Đã lưu kết quả vào {save_path}")
    else:
        st.sidebar.error("Không có dữ liệu kết quả để lưu.")
    # Save the DataFrame to an Excel file
    #result_excel_path = r'D:\Task\Du an PTM\Datasite-smartF\day_kwh_Longan_2024.xlsx'
    #result_df.to_excel(result_excel_path, index=False)
    st.markdown("**Bảng kết quả như sau:**")
    st.dataframe(result_df,hide_index=True)

# Function to view info the electricity in monthly
def month_view_kwh():
    global password
    if not MaKH or MaKH.strip() == "":
        st.warning("Vui lòng nhập đầy đủ thông tin Mã khách hàng (PB hoặc PK, 13 chữ) bên dưới")
        return
    
    st.empty()
    st.markdown(f"**Thông tin điện tiêu thụ theo tháng của trạm BTS có MaKH là {MaKH}, phản hồi từ web: https://cskh.evnspc.vn**")

    # Create JSON data
    json_data_common = {
        "MaDonViKH": "MOBIPHONEMLMN",
        "MaKhachHang": MaKH,
        "TenDangNhap": MaKH,
        "MatKhau": password,
        "MaChucNang": "HOADONCT",
        "Ky": "1",
        "Nam": Year
    }

    # Set up headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": "api.cskh.evnspc.vn",
        "User-Agent": "python-requests/2.26.0"
    }

    # Set up the URL
    url = "https://www.cskh.evnspc.vn/TraCuu/KhachHangTrongDiem"
    # Create an empty dataframe to store result
    result_df_split = pd.DataFrame()

    for mm in Month:
        json_data = {
            **json_data_common,
            "Thang": mm
        }
        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(json_data))
    
        # Khởi tạo biến tổng
        total_kwh = 0
        total_money = 0
        price_unit = ""

        # Check if the response is successful (status code 200)
        if response.status_code == 200:
            # Parse the response as JSON
            response_data = response.json()
            if response_data.get("Result") == "Error":
                # Display the error message
                st.error(response_data.get("Content"))
            elif response_data.get("Result") == []:
                # Wite note as 'content' in response
                st.write("Có phản hồi API nhưng không có giá trị trả về, giá trị là []")
            else:
                # Access the "Result" key and check if it exists
                result_list = response_data.get("Result", [])
                # Display the electriccity info of BTS
                #st.markdown("**Tháng " + str(mm) + " năm " + str(Year) + ":**")
                       
                # Lặp qua từng mục trong Result
                for entry in response_data["Result"]:
                    # Lấy giá trị ChiSoCu và ChiSoMoi, chuyển đổi thành số
                    chi_so_cu = float(entry["ChiSoCu"])
                    chi_so_moi = float(entry["ChiSoMoi"])

                    # Tính số kWh và cộng vào tổng
                    so_kwh = (chi_so_moi - chi_so_cu)*float(entry["HeSoNhan"])
                    total_kwh += so_kwh

                    # Lấy giá trị SoTien, chuyển đổi thành số
                    so_tien = float(entry["SoTien"])

                    # Cộng vào tổng số tiền
                    total_money += so_tien
                
                if total_kwh==0:
                    price_unit="Không có thông tin trả về, không tính được đơn giá"
                else:
                    price_unit=str(int(total_money/total_kwh))
                        
                formatted_money = "{:,.0f}".format(total_money)

                #st.write("Số kwh/tháng là:",int(total_kwh))
                #st.write("Số tiền/tháng là:",formatted_money)
                #st.write("Đơn giá điện là:",price_unit)
                #st.write("Chi tiết như sau:")
                #st.write(result_list)
                calculate_data = {
                    'kwh': total_kwh,
                    'money': formatted_money,
                    'price_unit': price_unit,
                }
                calculate_data = pd.DataFrame([calculate_data])

                # Using json_normalize to flatten the 'result' column                
                normalized_data = pd.json_normalize(response.json(),'Result')
                    
                # Concatenate along columns (axis=1)
                current_iteration_df = pd.concat([normalized_data,calculate_data], axis=1)

                # Concatenate along rows (axis=0)
                result_df_split = pd.concat([result_df_split, current_iteration_df], axis=0, ignore_index=True)

        else:
            st.error(f"Lỗi khi thực hiện request: {response.status_code}")
    
    if result_df_split is not None:        
        columns_to_drop = ["MaDonVi", "MaKhachHang", "SoCongTo"]
        for col in columns_to_drop:
            if col not in result_df_split.columns:
                return
            else:
                result_df_split = result_df_split.drop(["MaDonVi", "MaKhachHang", "SoCongTo"], axis=1)
                st.markdown("**Bảng kết quả như sau:**")
                st.dataframe(result_df_split,hide_index=True)
                
                # Chuyển đổi cột 'money' và 'Thang' sang số thực
                # Loại bỏ dấu phẩy từ các giá trị trong cột 'money'
                result_df_split['money'] = result_df_split['money'].str.replace(',', '')
                result_df_split['money'] = result_df_split['money'].astype(float)
                result_df_split['Thang'] = result_df_split['Thang'].astype(float)

                # Vẽ biểu đồ tiền điện theo tháng
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(result_df_split['Thang'], result_df_split['money'], marker='o', linestyle='-')
                ax.set_title(f'Biểu đồ số tiền theo tháng của MaKH: {MaKH}')
                ax.set_xlabel('Tháng')
                ax.set_ylabel('Số tiền')
                ax.tick_params(axis='x', rotation=0)  # Xoay nhãn trục x để dễ đọc hơn
                plt.tight_layout()
                st.pyplot(fig)
    
def month_get_kwh():
    global password
    # Record the start time
    start_time = time.time()
    # Đọc dữ liệu từ tệp Excel
    get_file_path = uploaded_file
    if get_file_path is None:
        st.sidebar.warning("Vui lòng tải file dữ liệu MaKH cần lấy thông tin kwh trước.")
        return
    
    st.sidebar.empty()
    st.sidebar.success("Đang thực hiện lấy thông tin điện ...")

    data = pd.read_excel(get_file_path)
    num_unique_MaKH = len(data[data["MaKH"].notnull()]["MaKH"].unique())
    num_rows = data.shape[0]
    num_rows_null=num_rows-num_unique_MaKH
    data = data.dropna(subset=['MaKH'])
    # Remove duplicates based on the specified column
    data = data.drop_duplicates(subset=['MaKH'])

    # Hiển thị thông tin DataFrame của file tải lên
    st.empty()
    st.markdown("**Thông tin của file MaKH tải lên:**")
    st.write(f"Có tổng cộng {num_rows} hàng. Trong đó:")
    st.write("- Số MaKH có dữ liệu hợp lệ (đã bỏ duplicate) là:",num_unique_MaKH)
    st.write("- Số MaKH có dữ liệu không hợp lệ là:",num_rows_null)

    # Common JSON data
    json_data_common = {
        "MaDonViKH": "MOBIPHONEMLMN",
        "MatKhau": password,
        "MaChucNang": "HOADONCT",
        "Ky": "1"
    }

    # Set up common headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": "api.cskh.evnspc.vn",
        "User-Agent": "python-requests/2.26.0"
    }

    # Set up the URL
    url = "https://www.cskh.evnspc.vn/TraCuu/KhachHangTrongDiem"

    # Create an empty list to store JSON responses
    json_responses = []
    
    # Create an empty dataframe to store result
    result_df_split = pd.DataFrame()
    
    # Tạo danh sách các tháng trong dãi đã chọn
    #selected_months = list(range(Month + 1))
    
    # Iterate over "PE" values in the DataFrame
    for pe_value in data['MaKH']:
        for yyyy in [Year]:
            for mm in Month:
                # Create JSON data for the specific combination
                json_data = {
                    **json_data_common,
                    "MaKhachHang": pe_value,
                    "TenDangNhap": pe_value,
                    "Thang": mm,
                    "Nam": yyyy
                }

                try:
                    # Make the POST request
                    response = requests.post(url, headers=headers, data=json.dumps(json_data))
                    
                    # Check if the response status is OK (200)
                    if response.status_code == 200:
                        # Append JSON response to the list
                        json_responses.append(response.json())
                    else:
                        print(f"Failed request for PE: {pe_value}, Year: {yyyy}, Month: {mm}. Status Code: {response.status_code}")
                except Exception as e:
                    print(f"Error during request for PE: {pe_value}, Year: {yyyy}, Month: {mm}. Error: {e}")
            
                # Create a DataFrame from the current iteration data
                current_iteration_data = {
                    'MaKH': pe_value,
                    'Year': yyyy,
                    'Month': mm,
                    'Note': []
                }
                current_iteration_df = pd.DataFrame([current_iteration_data])
            
                response_data = response.json()
                if response_data.get("Result") == "Error" :
                    # Wite note as 'content' in response
                    current_iteration_df['Note']=response_data.get("Content")
                    result_df_split = pd.concat([result_df_split, current_iteration_df], axis=0, ignore_index=True)
                elif response_data.get("Result") == []:
                    # Wite note as 'content' in response
                    current_iteration_df['Note']="Có phản hồi API nhưng không có giá trị trả về, giá trị là []"
                    result_df_split = pd.concat([result_df_split, current_iteration_df], axis=0, ignore_index=True)
                else:
                    #Wite note as 'OK'
                    current_iteration_df['Note']="OK"
                    # Khởi tạo biến tổng
                    total_kwh = 0
                    total_money = 0
                    price_unit = ""

                    # Lặp qua từng mục trong Result
                    for entry in response_data["Result"]:
                        # Lấy giá trị ChiSoCu và ChiSoMoi, chuyển đổi thành số
                        chi_so_cu = float(entry["ChiSoCu"])
                        chi_so_moi = float(entry["ChiSoMoi"])

                        # Tính số kWh và cộng vào tổng
                        so_kwh = (chi_so_moi - chi_so_cu)*float(entry["HeSoNhan"])
                        total_kwh += so_kwh

                        # Lấy giá trị SoTien, chuyển đổi thành số
                        so_tien = float(entry["SoTien"])

                        # Cộng vào tổng số tiền
                        total_money += so_tien
                    
                    formatted_money = "{:,.0f}".format(total_money)
                    if total_kwh==0:
                        price_unit="Không có thông tin trả về, không tính được đơn giá"
                    else:
                        price_unit=str(int(total_money/total_kwh))
                       
                    calculate_data = {
                    'kwh': total_kwh,
                    'money': formatted_money,
                    'price_unit': price_unit,
                    }
                    calculate_data = pd.DataFrame([calculate_data])

                    # Using json_normalize to flatten the 'result' column                
                    normalized_data = pd.json_normalize(response.json(),'Result')
                    
                    # Concatenate along columns (axis=1)
                    current_iteration_df = pd.concat([current_iteration_df, normalized_data,calculate_data], axis=1)

                    # Concatenate along rows (axis=0)
                    result_df_split = pd.concat([result_df_split, current_iteration_df], axis=0, ignore_index=True)
                    
    # Save the DataFrame to an Excel file
    if 'SiteID' in data.columns:
        result_df_split = pd.merge(result_df_split, data[['MaKH', 'SiteID']], on='MaKH', how='left')
        # Reorder columns dynamically
        column_order = ['SiteID'] + [col for col in result_df_split.columns if col != 'SiteID']
        result_df_split = result_df_split[column_order]
    num_unique_MaKH = len(result_df_split["MaKH"].unique())
    num_rows_OK = result_df_split["Note"].eq("OK").sum()
    num_rows_Error=num_unique_MaKH-num_rows_OK

    st.sidebar.empty()
    st.sidebar.success("Việc lấy thông tin đã hoàn thành.")
    st.empty()
    st.markdown("**Kết quả lấy thông tin điện tiêu thụ của trạm BTS, phản hồi từ web: https://cskh.evnspc.vn như sau**")
    st.write(f"Có tổng cộng {num_unique_MaKH} MaKH đã có phản hồi. Trong đó:")
    st.write("- Số MaKH có dữ liệu OK là:",num_rows_OK)
    st.write("- Số MaKH có dữ liệu lỗi là:",num_rows_Error)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Display the elapsed time
    st.markdown(f"**Thời gian thực hiện lấy thông tin: {elapsed_time:.2f} giây**")

    if result_df_split is not None:
        # Get the current date and time in a specific format
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Đường dẫn lưu trữ file mới trong thư mục Downloads trên ổ C        
        #month_str = "_".join(map(str, Month))
        save_path = os.path.join(os.path.expanduser("~"), "Downloads", f"{current_datetime}_result_get_kwh.xlsx")
        #save_path = os.path.join(os.path.expanduser("~"), "Downloads", f"result_get_kwh_{Year}_{month_str}.xlsx")
        #save_path = os.path.join(os.path.expanduser("~"), "Downloads", "result_get_kwh_{}_{}.xlsx".format(Year, month_str))

        # Lưu kết quả vào đường dẫn mới
        result_df_split.to_excel(save_path, index=False)
        st.sidebar.empty()
        st.sidebar.success(f"Đã lưu kết quả vào {save_path}")
    else:
        st.sidebar.error("Không có dữ liệu kết quả để lưu.")

def month_update_MaKH():
    # Đọc dữ liệu từ tệp Excel
    get_file_path = uploaded_file
    if get_file_path is None:
        st.sidebar.warning("Vui lòng tải file dữ liệu MaKH cần lấy thông tin kwh trước.")
        return
    
    st.sidebar.empty()
    st.sidebar.success("Đang thực hiện lấy thông tin điện ...")

    data = pd.read_excel(get_file_path)
    num_unique_MaKH = len(data[data["MaKH"].notnull()]["MaKH"].unique())
    num_rows = data.shape[0]
    num_rows_null=num_rows-num_unique_MaKH
    data = data.dropna(subset=['MaKH'])
    # Remove duplicates based on the specified column
    data = data.drop_duplicates(subset=['MaKH'])

    # Hiển thị thông tin DataFrame của file tải lên
    st.empty()
    st.markdown("**Thông tin của file MaKH tải lên:**")
    st.write(f"Có tổng cộng {num_rows} hàng. Trong đó:")
    st.write("-Số MaKH có dữ liệu hợp lệ là:",num_unique_MaKH)
    st.write("-Số MaKH có dữ liệu không hợp lệ (null-để trống) là:",num_rows_null)

    # Common JSON data
    json_data_common = {
        "MaDonViKH": "MOBIPHONEMLMN",
        "MatKhau": "Mbf#1234",
        "MaChucNang": "THEMKH",
        "Ky": "1"
    }

    # Set up common headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
        "Host": "api.cskh.evnspc.vn",
        "User-Agent": "python-requests/2.26.0"
    }

    # Set up the URL
    url = "https://www.cskh.evnspc.vn/TraCuu/KhachHangTrongDiem"

    # Tạo danh sách các tháng trong dãi đã chọn
    #selected_months = list(range(Month + 1))
    
    # Iterate over "PE" values in the DataFrame
    for pe_value in data['MaKH']:
        for yyyy in [Year]:
            for mm in Month:
                # Create JSON data for the specific combination
                json_data = {
                    **json_data_common,
                    "MaKhachHang": pe_value,
                    "TenDangNhap": pe_value,
                    "Thang": mm,
                    "Nam": yyyy
                }

                # Make the POST request
                response = requests.post(url, headers=headers, data=json.dumps(json_data))


# Sidebar for xử lý từ file excel cho nhiều trạm
# Sidebar
st.sidebar.title("Tab chức năng")
view_option = st.sidebar.radio("Chọn xem kwh:", ("Theo ngày", "Theo tháng"))
st.sidebar.header("Xem thông tin của nhiều MaKH đồng thời (qua file Excel)")
uploaded_file = st.sidebar.file_uploader("Vui lòng chọn file dữ liệu MaKH cần lấy thông tin điện (file excel, cột MaKH chứa Mã khách hàng)", type=["xlsx"])
# Dựa vào lựa chọn của người dùng, gọi hàm tương ứng
if view_option == "Theo ngày":
    # Main section for view info of BTS
    # Khai báo biến để nhập thông tin đầu vào trên giao diện web
    day_view_kwh_button = st.button("Xem thông tin 01 MaKH", on_click=day_view_kwh)
    st.subheader("Những thông tin cần nhập vào")
    MaKH = st.text_input("Mã khách hàng (PB hoặc PK, 13 ký tự)")
    from_ngay_mac_dinh = datetime(2024, 1, 1)
    to_ngay_mac_dinh = datetime(2024, 1, 7)
    From_date = st.date_input("Từ ngày",from_ngay_mac_dinh,format="DD/MM/YYYY")
    To_date = st.date_input("Đến ngày",to_ngay_mac_dinh,format="DD/MM/YYYY")
    day_get_kwh_button = st.sidebar.button("Lấy thông tin và lưu excel", on_click=day_get_kwh)
   
else:
    # Main section for view info of BTS
    # Khai báo biến để nhập thông tin đầu vào trên giao diện web
    month_view_kwh_button = st.button("Xem thông tin 01 MaKH", on_click=month_view_kwh)
    st.subheader("Những thông tin cần nhập vào")
    MaKH = st.text_input("Mã khách hàng (PB hoặc PK, 13 ký tự)")
    Year = st.selectbox("Năm", ['2024','2023', '2022', '2021','2020'])
    #Month = st.slider("Chọn dãi tháng", 1, 12, 1)
    Month = st.multiselect("Chọn tháng", ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],key="month_selector",default=['1'])
    month_get_kwh_button = st.sidebar.button("Lấy thông tin và lưu excel", on_click=month_get_kwh)
    #month_get_kwh_button = st.sidebar.button("Lấy thông tin và lưu excel", on_click=month_update_MaKH)
