# One_to_Multi_Representation(updating)

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

# Nov 1 updated
