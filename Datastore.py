db={"AG":[r"Deepface\AG\AG.jpg",
          r"Deepface\AG\panjabi_me.jpg",
          r"Deepface\AG\Self.jpeg",
          r"Deepface\AG\1.jpg",
          r"Deepface\AG\2.jpg",
          r"Deepface\AG\3.jpg",
          r"Deepface\AG\4.jpg",
          r"Deepface\AG\5.jpg",
          r"Deepface\AG\6.jpg",
          r"Deepface\AG\7.jpg",
          ],
    "MB":[r"Deepface\MB\MB.jpg",
          r"Deepface\MB\Acropolis.jpg",
          r"Deepface\MB\Rakhi.png"
          ],
    "DEBO":[r"Deepface\DEBO\DEBO.jpg",
            r"Deepface\DEBO\1.jpg",
            r"Deepface\DEBO\2.jpg",
            r"Deepface\DEBO\3.jpg",
            r"Deepface\DEBO\4.jpg",
            r"Deepface\DEBO\5.jpg",
            r"Deepface\DEBO\6.jpg",
            ],
    "DIPS":[r"Deepface\DIPS\Dips_pp.png"
            ],
    "MMM":[r"Deepface\MMM\SM1.jpg",
           r"Deepface\MMM\SM2.jpg",
           r"Deepface\MMM\SM3.jpg"]
    # "Crowd":[r"Deepface\CROWD\Crowd_SS.png",
    #          r"Deepface\CROWD\C_sample1.jpg"
    #         ]
    }
# print(db.values())
# print("\n")

def find_name(path):
    # print("ok")
    print(path) 
    for i in db.keys():
        if path in db[i]:
            return i
        # print(db[i])
    else:return 'Unknown' 
        
# print(find_name("Deepface\AG\3.jpg"))