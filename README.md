# One_to_Multi_Representation(updating...)

Images have not been widely used as input for reinforcement learning(RL) to control robots, mainly because: 
1. Most of the images currently used are 2-dimensional and therefore cannot effectively provide the relative positions of objects in space;
2. The redundant information in the image misleads the controller and slows down the training speed.

Therefore, we propose One-to-Multi (OTM) representation, which aims to derive the global representation of the environment by inputting only a single RGB image. Just like human eyes, when you see a scene from one angle, you can use your imagination to infer the perspective from multiple angles The time seriesl information in one epoch can be recorded by the OTM for the purpose of describing the current state more accurately. OTM is a highly efficient data format that improves the training speed and performance of RL algorithms. In addition, generalization ability allows it to be freely used in multiple unseened scenes in both simulators and reality.

# Introduction
RL is a promising control method, enabling robots to complete a series of tasks in the physical world. For instance, reaching, pushing, sliding and pick and place tasks developed by OpenAI can be solved using reinforcement learning algorithms with high accuracy. In this simulation environment, the observations required for training of the reinforcement learning algorithm can be extracted easily and accurately, which allows many of these  algorithms to be successful. However, in real world environments, it is very challenging to accurately extract observations.

Images have been used as input to the RL controller so that it can perceive the world and learn the dynamics of the environment. It makes the observation extraction more easily but reduces the performance of the controller compared with traditional data-based methods(such as taking robot data and target position from the environment). This is because most image-based RL algorithms use the autoencoding technology as the observation model or directly pass the images to the RL agent, the non-developed observation method reduces the quality of training.

In general, a large part of the input picture is occupied by redundant information, i.e. the background, noise, etc, these pixels contribute little to the improvement of the RL algorithm while they highly increase the computations needed. In addition, the input 2D pictures cannot accurately describe the 3D relative positions of objects in the environment. Fig.1 shows two views of a single scene that highlight this problem. Indeed, in Fig. 1a the robot has localized the target and is ready for grasping, however, the view in Fig. 1b shows that the gripper is not correctly positioned to grasp the ball. The left view will definitely mislead the agent. OTM is a solution to this problem as it can infer multiple -views of the real-time scene from a single image, hence providing more accurate 3D relative position.

As the robot moves in the environment, the later states of an epoch are highly dependent on previous states, and historical information plays a vital role in the evaluation of the states at a given point of time. For example, sometimes the camera will lose the target object because the position of the robot blocks the view of the object by the camera Directly passing a targetless image to the controller can lead to headless action and therefore no improvement. OTM can provide a more objective real-time estimation of the state of the environment as it uses both both previous states of the system as well as the current state of the system, which can solve the above-mentioned problems.
OTM has generalization for the same layout of different scene so that it can be used flexibly in the either simulator environment or reality

<img src="https://github.com/wq13552463699/One_to_Multi_Representation/blob/main/pictures/1.png" width="633" >

## Steps
### 1. Simulator set up
Pybullet is used as a physics engine to build a working platform for the UR5 robotic arm. The cameras are respectively placed on three key positions of 0°, -90°, and +90° on the X plane, pointing to the direction of the base of the robotic arm. In the training of the OTA model, the image captured by the 0° camera will be used as input to infer the images captured by the other two cameras. In order to develop the generalization of the OTM model, multiple environments with the same layout and different backgrounds will be created.
### 2. Real Robot arm environment
The working environment of the real robot is basically the same as the layout in the simulator. The UR5e robot is placed on the vention work platform with the Intel RealSense camera placed at 0° position on X plane and point to base.
### 3. Multi-view Infer
View inference will be implemented through stitching global and local views inferred in parallel. Local parts can strengthen the formation of the details, such as the pose of the robotic arm and the position of the target object, etc. Global part is to indicate the relative position and global frame. The model has the structure shown below.\
<img src="https://github.com/wq13552463699/One_to_Multi_Representation/blob/main/pictures/2.png" width="633" >
### 4. Time-Series data handle
In the control of the robot, the series of images input in a single epoch is strongly dependent on each other in time. We add GRU cells in the model to enable it to record historical information in the environment so that the current inference can be more accurate.
### 5. Principal component extraction
After the multi-view has been inferred to an acceptable level, the principal components of the original and the inferred images need to be extracted. They will be stacked together and VAE structure will be applied to reduce the data dimensionality so that extracting principal components. Since the reliability of the inferred images are lower than that of the original image, weight components will be added to make the calculation relatively biased toward the original image.
### 6. Generalizing
The purpose of generalizing the model is to maintain the same performance when the OTM model is put into a totally new scene. The scenes for training the OTM have the same layout but different color and components, it will be changed from time to time.
### 7. Apply OTM in RL
Place a camera at the 0° position of a scene that has a similar layout for training OTM, so that the image captured by the camera is the same as the image captured by the 0° camera in OTM's training scene. Compare the accuracy and training time of the OTM model as an observation model with that of the traditional observation model. Comparison: 1. OTM vs One raw image with AE. 2. OTM vs multi raw images with AE. 

## Logic Map
<img src="https://github.com/wq13552463699/One_to_Multi_Representation/blob/main/pictures/3.png" width="1000" >

## Installation
* Clone or download this repository to your local PC.
* The environment used in this experiment is in another folder of mine, please see:
https://github.com/wq13552463699/UR5E_robot_gym_env_Real_and_Sim/tree/main/Simulation
Please clone the content in this link, and put all the files together within this repository in you local PC

## Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
Please make sure to update tests as appropriate.


# Sep 28 updated
Developed the dataset and trained the RGAN model, which can accurately infer the image of another angle by observing the image of one angle. Please see:\
<img src="https://github.com/wq13552463699/One_to_Multi_Representation/blob/main/pictures/4.png" width="1000" >
### Loss Curve
<img src="https://github.com/wq13552463699/One_to_Multi_Representation/blob/main/pictures/6.png" width="1000" >

# Nov 1 updated
New RNN elements are introduced into the model, which enables the model to be able to process time-series data. But after adjusting the model structure and data set countless times, the overall results obtained are not ideal.\
The location of the target can be basically inferred. As shown in the following picture, the target in the input image was totally blocked by robot’s body, in theory, the pure GAN model cannot infer the unseen target. RNN was play a crucial role for solving this problem, it gives me some hope. The inferred side view image is not acceptable yet, but it can prove that the RNN can help on the prediction at least.
<img src="https://github.com/wq13552463699/One_to_Multi_Representation/blob/main/pictures/5.png" width="1000" >
### Loss Curve
<img src="https://github.com/wq13552463699/One_to_Multi_Representation/blob/main/pictures/7.png" width="1000" >


