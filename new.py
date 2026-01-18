students = []

search_Engine = [{"company":"google","location":"usa","type":"search engine"},
                {"company":"bing","location":"usa","type":"search engine"}] 

# for ser in search_Engine:


# for index, engine in enumerate(search_Engine):
#     print(index , engine["company"], engine["location"], engine["type"])


# for engine in search_Engine:
#     print(engine["company"], engine["location"], engine["type"])

for index in range(len(search_Engine)):
    engine = search_Engine[index]
    print(engine, index)
    # print(index , engine["company"], engine["location"], engine["type"])

for index, engine in enumerate(search_Engine):
    print(index,engine["company"], engine["location"], engine["type"])