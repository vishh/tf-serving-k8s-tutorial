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

"""A profiler client that prints statistics on the round trip serving time.

The profiler makes the same request as resnet_client.py, but computes the round
trip serving time for the request. You can set the number of times you would
like to send requests, and the profiler will compute various statistics on the
serving times, such as mean, median, min and max.
"""

from __future__ import print_function

import argparse

from PIL import Image
import time                                                
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
      help='Paths (local or url) to images you would like to label'
  )
  parser.add_argument(
      'images',
      type=str,
      nargs='+',
      help='Local paths to images you would like to label'
  )
  parser.add_argument(
      '-n',
      '--num_trials',
      type=int,
      default='.txt',
      help='File used to log batch serving request delays. Will create file'
           'if it does not exist. Otherwise, it will append to the file.'
  )
  args = parser.parse_args()

  images = args.images

  # Call the server num_trials times
  elapsed_times = []

  for t in range(0, args.num_trials):
    for image in images:
      # Each image needs to be 224x224x3 for ResNet.
      data = np.array(Image.open(image).resize((224, 224))).astype(np.float).reshape(-1, 224, 224, 3)
      np.set_printoptions(threshold=np.inf)                                                                                 
      json_request = '{{ "instances" : {} }}'.format(np.array2string(data, separator=',', formatter={'float':lambda x: "%.1f" % x}))
      url = "http://%s:%d/v1/models/%s:predict" %(args.server, args.port, args.model)
      start_time = int(round(time.time() * 1000))
      resp = requests.post(url, data=json_request)
      elapsed = int(round(time.time() * 1000)) - start_time
      print('Request delay: ' + str(elapsed) + ' ms')
      elapsed_times.append(elapsed)
      print('response.status_code: {}'.format(resp.status_code))

  print('Mean: %0.2f' % np.mean(elapsed_times))
  print('Median: %0.2f' % np.median(elapsed_times))
  print('Min: %0.2f' % np.min(elapsed_times))
  print('Max: %0.2f' % np.max(elapsed_times))

if __name__ == '__main__':
  main()
