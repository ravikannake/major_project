from __future__ import absolute_import, division, print_function

import argparse
import collections
import csv
import json
import math
import time
import re
from collections import OrderedDict
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont



def recognize_plate(fp,
                    regions=[],
                    key='ffc4d16cd7838d18c50d32169c56c10e24f664b6',
                    sdk_url=None,
                    config={},
                    camera_id=None,
                    timestamp=None,
                    mmc=None,
                    exit_on_error=True):
    data = dict(regions=regions, config=json.dumps(config))
 
    for _ in range(3):
      fp.seek(0)
      response = requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        files=dict(upload=fp),
        data=data,
        headers={'Authorization': 'Token ' + key})
        
      if response.status_code == 429:
        time.sleep(1)
      else:
        break

    if not response:
        return {}
    if response.status_code < 200 or response.status_code > 300:
        print(response.text)
        if exit_on_error:
            exit(1)
    return response.json(object_pairs_hook=OrderedDict)



