# OptimisedResNet
This Repository contains the experiments and implementation of optimized model of ResNet18
## Abstract
In the work shown in this report, we try to maximize the test accuracy of the ResNet18 model by keeping a constraint where the number of trainable parameters of the new ResNet18 architecture should not exceed 5 million. We experiment on the ResNet18 model by modifying its layer structure and subjecting it to various parameters and optimizers. Finally, inferring from the experimental results, we propose a modified architecture of ResNet18, which exhibits the highest test accuracy,  given the parameter constraint.
## Best Models
After trying out a various methodologies and combinations of optimizers, parameters, layer structures, it was found that 3 of the model configurations exhibited high test accuracy.
• Architecture 1: With Conv layers [2,1,1,1] block configuration for 64, 128, 256 & 512 channels respectively.
• Architecture 2: With 512 channel block removed with only [3,3,3] block configuration for 64, 128, 256 channels respectively, plus dropout for Conv layers and hidden linear layer.
• Architecture 3: Dropout out for the classification layer in both the above architecture

# Respository details
The BestArchitecture contains the model code used to setup and train the architecture along with the necessary plots. 
