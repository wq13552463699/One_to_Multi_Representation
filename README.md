# One_to_Multi_Representation

We describe an ONE-TO-MULTIPLE(OTM) image-based robotics environment representation method, which is to derive the global representation of the current scene based on taking a 2D image. The image of one perspective in the robot scene is used to infer images of other perspectives. The principal components of the original and inferred images are extracted and merged together to form new data. The trained OTM model will be used as the observation model of the reinforcement learning(RL) algorithm to solve a series of object manipulation tasks. The results will be compared with the current state-of-the-art image-based algorithms, our method will theoretically have higher efficiency and accuracy. Finally, the trained OTM and RL model will be transferred from the simulator to the real UR5E robotics system to solve the same tasks. Domain Randomization(DR) will be used to improve the generalization ability of the model, so that the trained model can have the same performance on the real robot as in the simulator.
