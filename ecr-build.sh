#!/bin/bash

set -o errexit
set -o xtrace

echo "Build AWS ECR"

BASE_IMAGE="r8.im/alexgenovese/train-sdxl-lora"

# echo "ignore all the small files, copy big files"
# find . -type f -size -10M > .dockerignore
# docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t base .

# BASE_ID=$(docker inspect $(BASE_IMAGE) --format='{{index .Id}}')

echo "ignore all the big files, copy copy files"
# find . -type f -size +10M > .dockerignore
docker build --build-arg BASE_IMAGE=$BASE_IMAGE -t reica-training .
docker tag reica-training:latest 209861239921.dkr.ecr.us-east-1.amazonaws.com/reica-training:latest
docker push 209861239921.dkr.ecr.us-east-1.amazonaws.com/reica-training:latest

echo "Completed"