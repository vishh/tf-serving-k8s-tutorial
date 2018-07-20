#!/usr/bin/env python2.7
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A client that talks to tensorflow_model_server loaded with an image model.

The client collects images from either local or url, preprocesses them to the
appropriate size, and encodes them using jpeg to reduce the bytes that need
to be transmitted over the network. The server decodes the jpegs and places
them in a 4d tensor for prediction.
"""

from __future__ import print_function

import argparse
from PIL import Image                                                           
import requests
import numpy as np

def main():
  # Command line arguments
  parser = argparse.ArgumentParser('Label an image using the cat model')
  parser.add_argument(
      '-s',
      '--server',
      help='URL of host serving the cat model'
  )
  parser.add_argument(
      '-p',
      '--port',
      type=int,
      default=8501,
      help='Port at which cat model is being served'
  )
  parser.add_argument(
      '-m',
      '--model',
      type=str,
      default='resnet',
      help='Model name'
  )
  parser.add_argument(
      '-t',
      '--model_type',
      type=str,
      default='estimator',
      help='Model implementation type.'
           'Default is \'estimator\'. Other options: \'keras\''
  )
  parser.add_argument(
      'images',
      type=str,
      nargs='+',
      help='Local paths to images you would like to label'
  )

  args = parser.parse_args()
  images = args.images
  for image in images:
    # Each image needs to be 224x224x3 for ResNet.
    imdata = Image.open(image).resize((224, 224))
    data = np.array(imdata).astype(np.float).reshape(-1, 224, 224, 3)
    print(len(data))
    np.set_printoptions(threshold=np.inf)                                                                                 
    json_request = '{{ "instances" : {} }}'.format(np.array2string(data, separator=',', formatter={'float':lambda x: "%.1f" % x}))
    url = "http://%s:%d/v1/models/%s:predict" %(args.server, args.port, args.model)
    resp = requests.post(url, data=json_request)
    print('response.status_code: {}'.format(resp.status_code))
    print('response.content: {}'.format(resp.content))
    
if __name__ == '__main__':
  main()
