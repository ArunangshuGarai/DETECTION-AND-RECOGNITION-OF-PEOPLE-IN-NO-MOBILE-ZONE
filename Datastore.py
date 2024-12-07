db={"ARUNANGSHU":[r"Deepface\AG\AG.jpg",
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
    "MOHINI":[r"Deepface\MB\MB.jpg",
          r"Deepface\MB\Acropolis.jpg",
          r"Deepface\MB\Rakhi.png"
          ],
    "DEBOJYOTI":[r"Deepface\DEBO\DEBO.jpg",
            r"Deepface\DEBO\1.jpg",
            r"Deepface\DEBO\2.jpg",
            r"Deepface\DEBO\3.jpg",
            r"Deepface\DEBO\4.jpg",
            r"Deepface\DEBO\5.jpg",
            r"Deepface\DEBO\6.jpg",
            ],
    "DIPTIMOY":[r"Deepface\DIPS\1.jpg",
                r"Deepface\DIPS\2.jpg",
                r"Deepface\DIPS\3.jpg",
                r"Deepface\DIPS\4.jpg",
                r"Deepface\DIPS\5.jpg"
            ],
    "SAGNIK":[r"Deepface\SAGNIK\1.jpg",
              r"Deepface\SAGNIK\2.jpg",
              r"Deepface\SAGNIK\3.jpg",
              r"Deepface\SAGNIK\4.jpg",
              r"Deepface\SAGNIK\5.jpg",
              r"Deepface\SAGNIK\6.jpg",
              r"Deepface\SAGNIK\7.jpg",              
              r"Deepface\SAGNIK\8.jpg",
              ],
    "DOGRA SIR":[r"Deepface\DOGRA SIR\1.jpg",
                 r"Deepface\DOGRA SIR\2.jpg",
                 r"Deepface\DOGRA SIR\3.jpg",
                 r"Deepface\DOGRA SIR\4.jpg",
                 r"Deepface\DOGRA SIR\5.jpg",
                 r"Deepface\DOGRA SIR\6.jpg",
                 r"Deepface\DOGRA SIR\7.jpg",
                 r"Deepface\DOGRA SIR\8.jpg"
                 ]
        }


def find_name(path):
    print(path) 
    for i in db.keys():
        if path in db[i]:
            return i
    else:return 'Unknown' 
        
