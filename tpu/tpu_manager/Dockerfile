# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory to /tpu_manager
WORKDIR /tpu_manager

# Copy only the necessary files into the container
COPY ./requirements.txt /tpu_manager/requirements.txt
COPY ./tpu_manager.py /tpu_manager/tpu_manager.py
COPY ./tpu_automate_training.py /tpu_manager/tpu_automate_training.py
COPY ./convert_to_records.py /tpu_manager/convert_to_records.py
COPY ./models/official/mnist /tpu_manager/models/official/mnist
COPY ./models/official/utils /tpu_manager/models/official/utils

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run app.py when the container launches
CMD ["python", "tpu_automate_training.py"]
