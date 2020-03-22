#-*- coding:utf-8 -*-
from selenium import webdriver
import os
import csv
from lxml import etree
import time
driver = webdriver.Chrome(executable_path="H:\\python3\\chromedriver_win32\\chromedriver.exe")
director = u"张艺谋"
file_name = os.path.join(".", director + ".csv")
base_url = 'https://search.douban.com/movie/subject_search?search_text=' + director + '&cat=1002&start='
out = open(file_name,'w', newline='', encoding='utf-8-sig')
csv_write = csv.writer(out, dialect='excel')

flags=[]
def download(request_url):
    driver.get(request_url)
    time.sleep(1)
    html = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
    html = etree.HTML(html)

    movie_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']//div[1]//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']")

    name_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']//div[1]//div[@class='item-root']/div[@class='detail']/div[@class='meta abstract_2']")
    num = len(movie_lists)
    if num > 15:
        movie_lists = movie_lists[1:]
        name_lists = name_lists[1:]
    for (movie, name_list) in zip(movie_lists, name_lists):
        if name_list.text is None:
            continue
        names = name_list.text.split("/")
        if names[0].strip() == director and movie.text not in flags:
            names[0] = movie.text
            flags.append(movie.text)
            csv_write.writerow(names)
    print('OK')
    print(num)
    if num > 14:
        return True
    else:
        return False

start = 0
while start < 10000:
    request_url = base_url + str(start)
    flag = download(request_url)
    if flag:
        start = start + 15
    else:
        break
out.close()
driver.close()
print("finished")
