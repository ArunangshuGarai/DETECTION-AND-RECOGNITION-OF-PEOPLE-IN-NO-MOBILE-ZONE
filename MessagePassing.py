import pushbullet
import datetime

# def pushmsg(area,msg):
def pushmsg(zone , name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Get your access token from Pushbullet.
    if name == 'ARUNANGSHU':
        access_token = "o.9jVdf5Bifd45YwTG5gu48bx7Rs4BMmlg"
    elif name == 'MOHINI':
        # access_token = "o.mfeafBhScjYBhfNsrDliU5bnvVQxXhZX"
        access_token = "o.9jVdf5Bifd45YwTG5gu48bx7Rs4BMmlg"
    elif name == 'DIPTIMOY':
        # access_token = "o.VhYrF0SjzrP1XfYdTIRcByTT52EHZVsG"
        access_token = "o.9jVdf5Bifd45YwTG5gu48bx7Rs4BMmlg"
    elif name == 'SAGNIK':
        # access_token = "o.PGo3YRaSKEc5qAbuZ2OcbYSqC5ptDeSx"
        access_token = "o.9jVdf5Bifd45YwTG5gu48bx7Rs4BMmlg"
    elif name == "AYUSH":
        # access_token = "o.VxpL0Uj0cSBjRTcPUD5VE5ctcMSKmE5n"
        access_token = "o.9jVdf5Bifd45YwTG5gu48bx7Rs4BMmlg"
    else: 
        print("Error in the device name")
    
    if name in ['ARUNANGSHU','MOHINI', 'DIPTIMOY', 'SAGNIK' , 'AYUSH' ]:  
        client = pushbullet.Pushbullet(access_token)
        client.push_note('Rail Ministry',f"{timestamp}: {name} fined ₹ 500 for using Phone in No-mobile Zone at Platform {zone}")
        # client.push_note(area,msg)
        print(f"Message Sent to {name}... ")

# pushmsg(datetime.datetime.now(),2,"ARUNANGSHU")
# # pushmsg("10:20 12-10-2024",2,"DIPTIMOY")
# # pushmsg("10:20 12-10-2024",2,"SAGNIK")
# pushmsg(datetime.datetime.now(),2,"MOHINI")
# pushmsg(datetime.datetime.now(),2,"AYUSH")