# RabbitMQ GPU Scheduler
The RabbitMQ GPU scheduler is a system that enables deep learning model experiments to be centrally managed through message brokers.

<br>

## Features

- Improve experimental productivity by managing multiple GPU servers through message brokers
- No message is lost even if the consumer or producer shuts down
- Prioritizing messages (model experiments)
- Send message processing logs and notification of errors to Discord Channel

<br>

## Getting Started : Broker Server
#### 1. install RabbitMQ server
```
apt-get update
apt-get install rabbitmq-server
```

#### 2. Base Settings
```
service rabbitmq-server start
rabbitmq-plugins enable rabbitmq_management

rabbitmqctl add_user admin 'password'
rabbitmqctl set_user_tags admin administrator
rabbitmqctl set_permissions -p / admin ".*" ".*" ".*"
rabbitmqctl delete_user guest
```
Create an account to use for the broker connection and delete the default guest account.  

<br>

## Getting Started : Producer Server
#### 1. clone this repository
#### 2. installing required libraries
``` bash
pip install pika
pip install python-dotenv
```
#### 3. in the .env file, enter the value of the variable (broker ip, port, etc)
#### 4. move to producer directory
#### 5. run producer.py
``` bash
python -B producer.py -c config.yaml -p 1
```
-c : files to send  
-p : message priority (default=0)

<br>

## Getting Started : Consumer Server
#### 1. clone this repository
#### 2. installing required libraries
``` bash
pip install pika
pip install python-dotenv
# In addition, libraries required for worker or model train
```
#### 3. in the .env file, enter the value of the variable (broker ip, port, etc)
#### 4. move to consumer directory
#### 5. run consumer.py (get message and run model trainer)
``` bash
python -B consumer.py
```
