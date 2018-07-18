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
import csv
import json
import time

from grpc.beta import implementations
import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import get_model_metadata_pb2
from google.protobuf import json_format

from image_processing import preprocess_and_encode_images

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
      default=9000,
      help='Port at which cat model is being served'
  )
  parser.add_argument(
      '-m',
      '--model',
      type=str,
      default='inception',
      help='Model name'
  )
  parser.add_argument(
      '-d',
      '--dim',
      type=int,
      default=224,
      help='Size of (square) image, an integer indicating its width and '
           'height. Resnet\'s default is 224'
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
      help='Paths (local, GCS, or url) to images you would like to label'
  )

  args = parser.parse_args()
  images = args.images

  # Convert image paths/urls to a batch of jpegs
  jpeg_batch = preprocess_and_encode_images(images, args.dim)

  # Call the server to predict top 5 classes and probabilities, and time taken
  result, elapsed = predict_and_profile(
      args.server, args.port, args.model, jpeg_batch)
  print(result)

def predict_and_profile(host, port, model, batch):

  # Prepare the RPC request to send to the TF server.
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'inception'
  request.model_spec.signature_name = 'predict_images'
  #try:
  result = stub.GetModelMetadata(mreq, 10.0)
  print(result)
 # except:
    #print("model not ready yet")
  request.inputs['images'].CopyFrom(
    tf.contrib.util.make_tensor_proto(batch[0], shape=[1]))
  # Call the server to predict, return the result, and compute round trip time
  start_time = int(round(time.time() * 1000))
  result = stub.Predict(request, 10.0)  # 60 second timeout
  elapsed = int(round(time.time() * 1000)) - start_time
  print(result)
  return result, elapsed

if __name__ == '__main__':
  main()
