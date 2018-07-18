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

"""A tensorflow serving health check module """

from __future__ import print_function

import argparse

from grpc.beta import implementations
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import get_model_metadata_pb2


def main():
  # Command line arguments
  parser = argparse.ArgumentParser('Check the health of tensorflow serving instance')
  parser.add_argument(
      '-s',
      '--server',
      type=str,
      default='localhost',
      help='URL of host serving the model'
  )
  parser.add_argument(
      '-m',
      '--model',
      type=str,
      default='inception',
      help='Model name'
  )
  parser.add_argument(
      '-p',
      '--port',
      type=int,
      default=9000,
      help='Port at which model is being served'
  )
  
  args = parser.parse_args()
  mreq = get_model_metadata_pb2.GetModelMetadataRequest()
  mreq.model_spec.name = args.model
  mreq.metadata_field.append('signature_def')
  channel = implementations.insecure_channel(args.server, int(args.port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  print(stub.GetModelMetadata(mreq, 10.0))
      
if __name__ == '__main__':
  main()
