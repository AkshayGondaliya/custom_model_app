**Steps to run this Repository on ubuntu 22.04 system **

**My System configuration**

os:-Linux, ubuntu-22.04

RAM:- 8GB

Processor :- i5, 8th Gen.

**Dependency guide**

- Please ensure that your docker desktop is installed by running bellowe command.
`docker` 

- if it shows "Command 'docker' not found", then install docker destop from bellow given link.
[install docker](https://docs.docker.com/engine/install/ubuntu/)

- command to chek whether your docker service is running or not :- 
`sudo systemctl is-active docker` 

- If it is "inactive" then start by running bellow given command.
`sudo systemctl start docker` 

**Clone The Repository**

`https://github.com/AkshayGondaliya/custom_model_app.git`

**+++++++++++  Deployment process for dockerized flask application  +++++++++++**

`cd custom_model_app/flaskapp` 

**1> Build docker image using the command given below**

`docker build -t myflaskapp .`

After successful execution of the above given command we will have docker one new docker image "myflaskapp".

**2> check image using command**

`sudo docker images`

**3> Now, Let's build a docker container from "myflaskapp" image**

`sudo docker run -d --name object_detection -p 5000:5000 myflaskapp`

**4> After success full execution of the above mentioned command  you can check your running application using the link**

`http://127.0.0.1:5000`

**5> How to use app**

step 1:- sign up

step 2:- select "1(upload image)" from "Enter your choice" option.

srep 3:- upload image 

step 4:- Click on submit button.

finally you can see your result on new arrived screen.


**6> stop your app using command**

`sudo docker stop object_detection`

**7> To start app again**

`sudo docker start object_detection`

**+++++  all model training , evaluation ,testing and saving process  +++++**

    Used Dataset -> cifar-10

    Dataset source -> [tensorflow cifar-10](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data)

    number of classes classes -> 10  ["airplabe","automobile","bird","cat","deer","dog","frog","horse","ship","Truck"]

    number of training images -> 50000

    number of testing images -> 10000

    Dimention of the images -> (32,32,3)

- go to `cd custom_model_app/model_building_and_evaluation`

- directory 1> custom_vs_resnet
      
    Here we have compared custom trained model and re-trained resnet-50 .

    you can also find accuracy and loss comparision graphs image

- directory 2> kfold_custom_vs_tuner_custom
      
    Here we have compared custom kfold trained model and hyperparameter optimized ketas tuner model .

    you can also find accuracy and loss comparision graphs image



** Idea on further improvement of custom object detection model **

- since my system property (i5, 8GB RAM) was bottleneck for me to train deep neural network, I was able to rech 70 % accuracy, but with good GPU system we can 
  try below mentioned option for model accuracy gain.  

1:- we Can add more  number of Convolutional layes followed by MaxPooling layers for better feature extraction. 

2:- If can build model of more than 10 layers of Covolutional layers with dropout layers after maxpooling layer to avoid the problem of overfitting and also help in reduction of model training time.

3:- we can train our model on multiple epochs(>20 or >30) for better feature learning(filter trainning).

4:- we can apply data augmentation techniques such as random rotation, flipping, cropping, or zooming to increase the diversity of the training data. This step helps the model generalize better to unseen data and reduces overfitting.

- Thank you
