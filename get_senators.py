import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
from datetime import datetime
import csv

# os.environ['MOZ_HEADLESS'] = '1'  # run on the background
browser = webdriver.Firefox(executable_path='/usr/local/bin/geckodriver')
print(browser.session_id)

browser.get("http://www.parliament.go.ke/the-senate/senators")
tabsel = ".cols-6"
table = browser.find_element_by_css_selector(tabsel)
theader = table.find_element_by_tag_name(
    "thead").find_element_by_tag_name("tr")

csv_headers = [td.text for td in theader.find_elements_by_tag_name("th")[:-1]]
csv_headers[1] = "Photo"
with open("Senators.csv", "w") as fp:
    writer = csv.writer(fp)
    writer.writerow(csv_headers)
    print(",".join(csv_headers))
    for i in range(7):
        tbody = browser.find_element_by_css_selector(".cols-6").find_element_by_tag_name(
            "tbody").find_elements_by_tag_name("tr")
        for tr in tbody:
            WebDriverWait(browser, 3).until(
                EC.presence_of_element_located((By.TAG_NAME, "table"))
            )
            tds = [td.text for td in tr.find_elements_by_tag_name("td")[:-1]]
            tds[1] = tr.find_elements_by_tag_name("td")[1].find_element_by_tag_name(
                "img").get_attribute("src").split("?")[0]
            # replace photo with photo_url
            writer.writerow(tds)
            print(
                ",".join(tds))
        next_page = browser.find_element_by_css_selector(
            "li.pager__item.pager__item--next")
        next_page.click()
        sleep(2)
