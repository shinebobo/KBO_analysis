from selenium import webdriver as wb
from selenium.webdriver.support.select import Select
import time
from bs4 import BeautifulSoup as bs
import pandas as pd
import os

def change_mode(driver):
    # 리그를 정규 리그로 // 한 번 클릭으로는 에러가 있어서 2번 클릭을 통해서 로드해줌.
    mode_table=driver.find_element_by_id("cphContents_cphContents_cphContents_ddlSeries_ddlSeries")
    league_mode_selector=Select(mode_table)
    league_mode_selector.select_by_value("7")
    mode_table.click()
    mode_table2=driver.find_element_by_id("cphContents_cphContents_cphContents_ddlSeries_ddlSeries")
    league_mode2_selector=Select(mode_table2)
    league_mode2_selector.select_by_value("0")
    mode_table2.click()
    time.sleep(1)
    
def get_year_list(driver):
    # 년도 설정
    year_table = driver.find_element_by_id("cphContents_cphContents_cphContents_ddlSeason_ddlSeason")
    year_selector=Select(year_table)
    year_list = [x.get_attribute("value") for x in year_table.find_elements_by_tag_name("option")]
    return year_list

def set_year(dirver, year):
    year_table = driver.find_element_by_id("cphContents_cphContents_cphContents_ddlSeason_ddlSeason")
    year_selector=Select(year_table)
    year_selector.select_by_value(year)
    time.sleep(1)

def crolling_info(driver, year):
    html_source = driver.page_source
    soup = bs(html_source, 'html.parser')
    table = soup.find('table', {'class':'tData tt'})

    index_list = [] ## 타격에 대한 index들을 저장합니다.
    data_list = []

    for ind in soup.find_all('th'):
        index_list.append(ind.get_text())

    col_tag = table.find_all('td')
    num = int((len(col_tag)+1)/len(index_list) - 1) ## 합계 칸이 존재하므로 -1을 해줘야함

    idx = 0
    temp_num = 0
    data = {}
    data["년도"] = year
    for col in col_tag:
        idx = idx % len(index_list)
        data[index_list[idx]] = col.get_text()
        idx+=1

        if idx == len(index_list):
            temp_num += 1
            data_list.append(data)
            data = {}
            data["년도"] = year
            if temp_num == num:
                break
    return data_list

driver = wb.Chrome(executable_path="C:\\Users\\shine\\Downloads\\chromedriver_win32\\chromedriver.exe")
# url = "https://www.koreabaseball.com/Record/Team/Pitcher/Basic1.aspx"
url_base = "https://www.koreabaseball.com/Record/Team"
url_list = ["Hitter/BasicOld.aspx", "Pitcher/BasicOld.aspx", "Defense/Basic.aspx", "Runner/Basic.aspx"]
csv_name_list = ["Hitting.csv", "Pitching.csv", "Defensing.csv", "Running.csv"]
# driver.get(url)

# for part, name in zip(url_list, csv_name_list):

part = url_list[2]
name = csv_name_list[2]

driver.get(os.path.join(url_base, part))
change_mode(driver)
year_list = get_year_list(driver)
total_data = []
for year in year_list:
    set_year(driver, year)
    data_list = crolling_info(driver, year)
    time.sleep(0.5)
    try:
        more_record = driver.find_element_by_class_name("more_record")
        next_button = more_record.find_element_by_class_name("next")
        next_button.click()
        more_data_list = crolling_info(driver, year)
        for data1, data2 in zip(data_list, more_data_list):
            data1.update(data2)
        more_record2 = driver.find_element_by_class_name("more_record")
        prev_button = more_record2.find_element_by_class_name("prev")
        prev_button.click()
    except:
        print("NO_more_record")
    total_data = total_data + data_list

pd.DataFrame(total_data).set_index('팀명').to_csv(name)
print("END!")
    
