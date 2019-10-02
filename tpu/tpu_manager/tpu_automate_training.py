#!/usr/bin/env python

# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
A script that creates and manages several Cloud Platform resources that
are required to run training on Cloud TPUs.
This script relies on tpu_manager.py for several methods that simplify
API calls to Cloud Storage, Compute Engine, IAM, and Cloud TPU APIs.
'''
import uuid
import os
import subprocess
import gzip
import shutil
import logging
import googleapiclient.discovery
from distutils.util import strtobool
from google.cloud import storage
import tpu_manager # Requires tpu_manager.py in the same CWD.

# Compute and TPU variables. If you do not have environment variables
# set on the system or container, the script uses the values that you
# specify here.
PROJECT = os.environ.get('GCLOUD_PROJECT', 'my-project')
NETWORK = os.environ.get('NETWORK', 'default')
ZONE = os.environ.get('ZONE', 'us-central1-c')
TPU_TYPE = os.environ.get('TPU_TYPE', 'v2-8')
FRAMEWORK = os.environ.get('FRAMEWORK', '1.14')
JOB_ID = os.environ.get('JOB_ID', '{project}-tpu-{uid}'.format(
    project=PROJECT, uid=str(uuid.uuid4())))
PREEMPTIBLE_TPU = strtobool(os.environ.get('PREEMPTIBLE_TPU', 'False'))
RESERVED_TPU = strtobool(os.environ.get('RESERVED_TPU', 'False'))
TPU_ADDRESS = os.environ.get('TPU_ADDRESS', 'None')

# Cloud Storage variables
STORAGE_LOCATION = os.environ.get('STORAGE_LOCATION', 'us-central1')
PREPROCESS = strtobool(os.environ.get('PREPROCESS', 'True'))
DATA_DIR = os.environ.get('DATA_DIR', 'data/')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output/')

# Change CORE_RATIO only if the number of cores for each TPU IP address
# changes. CIDR range size is 33-(max(8, cores)//CORE_RATIO).bit_length())
CORE_RATIO = 4

# Model variables specificlly for MNIST
ITERATIONS = os.environ.get('ITERATIONS', '4000')
TRAIN_STEPS = os.environ.get('TRAIN_STEPS', '10000')

# [START run_command_local]
def execute(cmd, cwd=None, capture_output=False, env=None, raise_errors=True):
    """Execute an external command (wrapper for Python subprocess)."""
    logging.info('Executing command: {cmd}'.format(cmd=str(cmd)))
    stdout = subprocess.PIPE if capture_output else None
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stdout)
    output = process.communicate()[0]
    returncode = process.returncode
    if returncode:
        # Error
        if raise_errors:
            raise subprocess.CalledProcessError(returncode, cmd)
        else:
            logging.info('Command returned error status %s', returncode)
    if output:
        logging.info(output)
    return returncode, output
# [END run_command_local]

# [START preprocess_mnist]
def preprocess_mnist():
    '''
    Define the steps for pre-processing data before training your model.
    '''
    execute([
        'python3',
        './convert_to_records.py',
        '--directory=./{path}'.format(path=DATA_DIR)])

    # Unzip the `.gz` files created by the pre-processing script.
    for root, subdirs, files in os.walk('./{path}'.format(path=DATA_DIR)):
        for file in files:
            if file.endswith(".gz"):
                p_in = os.path.join(root, file)
                p_out = os.path.splitext(p_in)[0]
                with gzip.open(p_in, 'rb') as f_in, open(p_out, 'wb') as f_out:
                    print('Extracted {p_in} to {p_out}'.format(
                        p_in=p_in,
                        p_out=p_out))
                    shutil.copyfileobj(f_in, f_out)
                    f_in.close()
                    f_out.close()
                    os.remove(p_in)
# [END preprocess_mnist]

# [START train_mnist]
def train_mnist():
    '''
    Define the steps for training your model on a Cloud TPU. This example
    starts a subprocess to run the `mnist_tpu.py` script. Optionally you can
    import your TF module and run it within Python.
    '''
    execute([
        'python3',
        './models/official/mnist/mnist_tpu.py',
        '--tpu={tpu_name}'.format(tpu_name=JOB_ID),
        '--data_dir=gs://{bucket}/{path}'.format(
            bucket=JOB_ID,
            path=DATA_DIR),
        '--model_dir=gs://{bucket}/{path}'.format(
            bucket=JOB_ID,
            path=OUTPUT_DIR),
        '--iterations={iterations}'.format(iterations=ITERATIONS),
        '--train_steps={train_steps}'.format(train_steps=TRAIN_STEPS),
        '--tpu_zone={zone}'.format(zone=ZONE),
        '--gcp_project={project}'.format(project=PROJECT),
        '--use_tpu=True'
        ])
# [END train_mnist]

def main():
    '''
    Main process to provision Cloud Platform resources, preprocess the
    training data, start a TPU node, and run the training script.
    '''

    # Preprocess the data if necessary before configuring resources and
    # starting training.
    if PREPROCESS:
        try:
            preprocess_mnist()
        except Exception:
            raise Exception('Preprocessing failed. Aborting training.')

    # [START cloud_platform_steps]
    # Initialize the necessary client libraries.
    tpu = googleapiclient.discovery.build(
        'tpu', 'v1', cache_discovery=False)
    compute = googleapiclient.discovery.build(
        'compute', 'v1', cache_discovery=False)
    # The Storage client is handled by importing google.cloud.storage

    # Create a new Cloud Storage bucket.
    try:
        bucket = tpu_manager.create_bucket(
            storage,
            JOB_ID,
            location=STORAGE_LOCATION)
    except Exception:
        raise Exception('Could not create the bucket for this training job.')

    # Upload the prepared data files to the Cloud Storage bucket.
    try:
        tpu_manager.upload_dir(bucket, './{path}'.format(path=DATA_DIR))
    except Exception:
        raise Exception('Could not upload training data to the bucket.')

    # Reserve a CIRD range for the TPU node. The tpu_manager.reserve_cidr()
    # method automatically finds an open IP address range of the appropriate
    # size for your TPU node and reserves it.
    try:
        cidr = tpu_manager.reserve_cidr(
            compute,
            PROJECT,
            NETWORK,
            JOB_ID,
            33-(max(8, int(TPU_TYPE.split('-')[1])//CORE_RATIO).bit_length()),
            TPU_ADDRESS)
    except Exception:
        raise Exception('Could not reserve a CIDR for this TPU node.')

    # Start the TPU node right before you submit the job.
    tpu_node = tpu_manager.create_tpus(
        tpu, PROJECT, JOB_ID, NETWORK, ZONE, TPU_TYPE, FRAMEWORK, cidr,
        preemptible=PREEMPTIBLE_TPU, reserved=RESERVED_TPU
    )

    # Grant the TPU read access to your Cloud Storage bucket.
    tpu_manager.tpu_bucket_access(
        bucket,
        tpu_node['serviceAccount'],
        'roles/storage.objectAdmin')
    # [END cloud_platform_steps]

    # [START training]
    # Start the training process now that the resources are in place.
    try:
        train_mnist()
    except Exception as e:
        logging.exception(e)
    # [END training]

    # [START cleanup]
    # Clean up Cloud Platform resources to reduce costs.
    print('Cleaning up unused resources.')
    tpu_manager.delete_tpus(tpu, PROJECT, ZONE, JOB_ID)
    tpu_manager.release_cidr(compute, PROJECT, JOB_ID)

    # If your application needs to deploy the trained model immediately,
    # you can download the results of this training run to a local directory.
    # Alternatively, another application can read the results from the bucket.
    tpu_manager.download_blobs(storage, bucket.name, OUTPUT_DIR,
                               'results-{id}'.format(id=JOB_ID))

    # Delete the `DATA_DIR` and keep the `OUTPUT_DIR` with the results for
    # another application that needs those results to run inference processes.
    tpu_manager.delete_blobs(storage, bucket.name, DATA_DIR)
    print('Model results are still available in {bucket}/{path}'.format(
        bucket=bucket.name,
        path=OUTPUT_DIR))

    # Optionally delete the entire Cloud Storage bucket and all contents.
    # tpu_manager.delete_bucket(storage, bucket.name)
    # [END cleanup]

if __name__ == '__main__':

    main()
