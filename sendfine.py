import requests
import json



def send_fine(mobile_num):

    url = "https://www.fast2sms.com/dev/bulk"


    my_data = {

    # default Sender ID
    'sender_id': 'FSTSMS',

    'message': "This is to inform that you have violated red signal at xyz junction at 12:30pm and you have to pay \n"
               "fine of Rs. 500 to nearest traffic police station or through online mode\n.With regards\n,"
               "Nagpur Traffic Police",

    'language': 'english',
    'route': 'p',

    'numbers': mobile_num

    }

    headers = {
    'authorization': 'KqTdIWM7oRA6Uf4lEuNxSwgrp1bPsQ0BvYjc3h5VCz2kFiHe8alEWbZkVe0N62YH5DU3xKdaoPt4vO1G',
    'Content-Type': "application/x-www-form-urlencoded",
    'Cache-Control': "no-cache"

    }


    # make a post request
    response = requests.request("POST",url,data = my_data,headers = headers)

    #load json data from source
    returned_msg = json.loads(response.text)

    # print the send message
    print(returned_msg['message'])


    return returned_msg['message'][0]

