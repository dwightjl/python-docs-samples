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
Several methods to simplify creating and managing Cloud Platform resources.

Many of these methods require initialized objects for Google Cloud APIs in the
googleapiclient.discovery. For example:

tpu = googleapiclient.discovery.build(
    'tpu', 'v1', cache_discovery=False)
compute = googleapiclient.discovery.build(
    'compute', 'v1', cache_discovery=False)
from google.cloud import storage
'''

from time import sleep
import os

# [START create_bucket]
def create_bucket(storage, name, location='us-central1',
                  storage_class='REGIONAL'):
    '''Create a GCS bucket.'''
    client = storage.Client()
    bucket = storage.Bucket(client, name)
    bucket.location = location
    bucket.storage_class = storage_class
    bucket.iam_configuration.bucket_policy_only_enabled = True
    client.create_bucket(bucket)
    return bucket
# [END create_bucket]


# [START delete_bucket]
def delete_bucket(storage, bucket_name):
    '''Deletes a bucket and all of its contents.'''
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    delete_blobs(storage, bucket_name, '', delete_all=True)
    bucket.delete()
    print('Bucket {bucket} deleted'.format(bucket=bucket.name))
# [END delete_bucket]


# [START delete_blobs]
def delete_blobs(storage, bucket_name, prefix, delete_all=False):
    '''Delete one or more blobs from a Cloud Storage bucket that
    have a specified prefix. Require delete_all=True to delete all files
    in the bucket, which prevents accidental deletions due to empty prefix
    values.'''
    if prefix == '' and delete_all == False:
        raise ValueError('You must specify both prefix='' and '
                         'delete_all=True to delete all objects from '
                         'the bucket.')
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    print('Deleting all files from {bucket} with prefix {prefix}'.format(
        bucket=bucket_name,
        prefix=prefix))
    for blob in blobs:
        blob.delete()
        print('Deleted: {blob}'.format(blob=blob.name))
# [END delete_blobs]


# [START tpu_bucket_access]
def tpu_bucket_access(bucket, sa_email, role):
    '''Grant the TPU service account access to a Cloud Storage bucket.'''
    policy = bucket.get_iam_policy()
    policy[role].add('serviceAccount:{sa_email}'.format(sa_email=sa_email))
    bucket.set_iam_policy(policy)
    print('Granted the {} role to {} on {}.'.format(
        role,
        sa_email,
        bucket))
# [END tpu_bucket_access]


# [START upload_dir]
def upload_dir(bucket, source, folder=''):
    '''Uploads a directory to a GCS bucket bucket.'''
    for root, subdirs, files in os.walk(source):
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(folder, root.strip('./'), file)
            blob = bucket.blob(destination_path)
            blob.upload_from_filename(source_path)
            print('File {} uploaded to {}.'.format(
                source_path, destination_path))
# [END upload_dir]


# [START download_blobs]
def download_blobs(storage, bucket_name, prefix, local_dir=os.getcwd()):
    '''Download one or more blobs from a Cloud Storage bucket that have a
    specified prefix. This method does overwrite existing files.'''
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    print('Downloading all files from {bucket} with prefix {prefix}'.format(
        bucket=bucket_name,
        prefix=prefix))
    for blob in blobs:
        filepath = os.path.join(local_dir, blob.name)
        os.makedirs(filepath.rsplit('/', 1).pop(0), exist_ok=True)
        # If the path ends in '/', it is a folder. os.makedirs() already
        # created the path. Otherwise, download the object.
        if not blob.name.endswith('/'):
            try:
                blob.download_to_filename(filepath)
                print('Downloaded: {file}'.format(file=blob.name))
            except (IsADirectoryError, NotADirectoryError):
                raise Exception('Cannot download the contents of bucket '
                                '{bucket}. Unix file systems do not allow '
                                'files and folders to have the same name. '
                                'Change your bucket object names or folder '
                                'names to be Unix compatible.'.format(
                                    bucket=bucket_name))
# [END download_blobs]


# [START reserve_cidr]
def reserve_cidr(compute, project, network, name, size, address=None):
    '''Reserve a specific range of addresses based on the number of TPU cores
    needed. The TPU API does not actually use the globalAddress object, but
    the method automatically identifies an open range on your network for
    your TPU node.'''

    cidr_config = {
        'network': '/global/networks/{network}'.format(
            network=network
        ),
        'name': name,
        'purpose': 'VPC_PEERING',
        'addressType': 'INTERNAL',
        'prefixLength': size
    }

    # If the method receives a manual address, append it to the cidr_config.
    # Otherwise, the system will automatically identify an open address range.
    if address:
        cidr_config.update({'address': address})

    # Reserve an address range for the new TPU node.
    # Reserved addresses are not required for TPUs, but they help to
    # identify open CIDR blocks automatically.
    operation = compute.globalAddresses().insert(
        project=project,
        body=cidr_config).execute()

    while True:
        print('Waiting for address block to allocate.')
        sleep(1)
        status = compute.globalOperations().get(
            project=project,
            operation=operation['name']).execute()
        if 'error' in status:
            raise Exception(status['error'])
        if status['status'] == 'DONE':
            break

    # Get the address range that Compute Engine Networking reserved.
    return compute.globalAddresses().get(
        project=project,
        address=name,
        fields='address,prefixLength'
        ).execute()
# [END reserve_cidr]


# [START release_cidr]
def release_cidr(compute, project, name):
    '''Release a reserved CIDR range that a TPU node no longer requires.'''

    operation = compute.globalAddresses().delete(
        project=project, address=name).execute()

    while (compute.globalOperations().get(
            project=project,
            operation=operation['name']).execute()['status'] != 'DONE'):
        print('Waiting for address block to release.')
        sleep(1)
# [END release_cidr]


# [START create_tpus]
def create_tpus(
        tpu, project, name, network, zone, tpu_type,
        tf_version, cidr, preemptible=False, reserved=False
    ):

    '''Create a TPU node with specific properties including a TPU type and
    the TensorFlow version that the TPU node must be compatible with. Pass
    preemptible=True to make the TPU node preemptible, or reserved=True
    if you want this TPU node to use your committment quota.'''

    tpu_config = {
        'acceleratorType': tpu_type,
        'tensorflowVersion': tf_version, 'network': network,
        'cidrBlock': '{address}/{prefix}'.format(
            address=cidr['address'], prefix=cidr['prefixLength']),
        'schedulingConfig': {
            'preemptible': preemptible,
            'reserved': reserved
        }
    }

    try:
        operation = tpu.projects().locations().nodes().create(
            parent='projects/{project}/locations/{zone}'.format(
                project=project,
                zone=zone),
            nodeId=name,
            body=tpu_config).execute()
    except Exception:
        raise Exception('Could not create a new TPU device.')

    while not (tpu.projects().locations().operations().get(
            name=operation['name']).execute()['done']):
        print('Waiting for TPU to start.')
        sleep(30)

    return tpu.projects().locations().nodes().get(
        name='projects/{project}/locations/{zone}/nodes/{name}'.format(
            project=project,
            zone=zone,
            name=name
        )).execute()
# [END create_tpus]


# [START delete_tpus]
def delete_tpus(tpu, project, zone, name):
    '''Delete a TPU node.'''

    operation = tpu.projects().locations().nodes().delete(
        name='projects/{project}/locations/{zone}/nodes/{name}'.format(
            project=project, zone=zone, name=name
        )).execute()

    while not (tpu.projects().locations().operations().get(
            name=operation['name']).execute()['done']):
        print('Waiting for TPU {name} to delete.'.format(name=name))
        sleep(30)
# [END delete_tpus]
