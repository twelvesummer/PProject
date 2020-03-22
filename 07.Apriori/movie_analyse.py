#-*- conding:utf-8 -*-
import csv
import os
from efficient_apriori import apriori
director = u'张艺谋'
file_name = os.path.join(".", director+".csv")
lists = csv.reader(open(file_name,'r',encoding='utf-8-sig'))
data = []
for names in lists:
    name_list = []
    for name in names:
        name_list.append(name.strip())
    if len(name_list) > 1:
        data.append(name_list[1:])
print(data)
itemsets,rules = apriori(data, min_support=0.05, min_confidence=1)
print(itemsets)
print(rules)
