###Steps to run this Repository on ubuntu 22.04 system.

### My System configuration.

os:-Linux, ubuntu-22.04

RAM:- 8GB

Processor :- i5, 8th Gen.

### Dependency guide ..

- Please ensure that your docker desktop is installed by running bellowe command.
`docker` 

- if it shows "Command 'docker' not found", then install docker destop from bellow given link.
[install docker](https://docs.docker.com/engine/install/ubuntu/)

- command to chek whether your docker service is running or not :- 
`sudo systemctl is-active docker` 

- If it is "inactive" then start by running bellow given command.
`sudo systemctl start docker` 

###  Clone The Repository:

`https://github.com/AkshayGondaliya/custom_model_app.git`

`cd custom_model_app` 

###  Build docker image using the command given below

`docker build -t myflaskapp .`

After successful execution of the above given command we will have docker one new docker image "myflaskapp".

###  check image using command..

`sudo docker images`

### Now, Let's build a docker container from "myflaskapp" image  ..

`sudo docker run -d --name object_detection -p 5000:5000 myflaskapp`

### After success full execution of the above mentioned command  you can check your running application using the linkk:-

`http://127.0.0.1:5000`

###  How to use app:

step 1:- sign up

step 2:- select "1(upload image)" from "Enter your choice" option.

srep 3:- upload image 

step 4:- Click on submit button.

finally you can see your result on new arrived screen.

-Thank you

###  stop your app using command :-

`sudo docker stop object_detection`

###  To start app again :-

`sudo docker start object_detection`

### Idea on further improvement of custom object detection model.

1:- we Can add more  number of Convolutional layes followed by MaxPooling layers for better feature extraction. 

2:- If can build model of more than 10 layers of Covolutional layers with dropout layers after maxpooling layer to avoid the problem of overfitting and also help in reduction of model training time.

3:- we can train our model on multiple epochs(>20 or >30) for better feature learning(filter trainning).

4:- we can apply data augmentation techniques such as random rotation, flipping, cropping, or zooming to increase the diversity of the training data. This step helps the model generalize better to unseen data and reduces overfitting.