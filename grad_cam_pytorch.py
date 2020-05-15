import torch
import cv2
import numpy as np
from PIL import Image

class Torch_GradCam:
    hook_handles = []
    hook_g = None
    def __init__(self, model, target_layer):
        """
        Pytorch class implementation of Grad-CAM
        args:
            model: pre-trained model.
            target_layer: (str) type of the target layer that you want to visualize the activation map from.
        """
        self.model = model.eval()
        self.target_layer = target_layer
        self.class_output = None
        self.hook_handles.append(self.model._modules.get(target_layer).register_backward_hook(self._hook_g))

    def _hook_g(self, module, input, output):
        """
        Gradient hook
        Ref: https://blog.paperspace.com/pytorch-hooks-gradient-clipping-debugging/
        """
        self.hook_g = output[0].data

    def get_activations(self,x):
        """
        return the activation map of the selected target layer
        """
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name == self.target_layer:
                #feature map of the selected target layer
                return x

    def forward_pass(self,x):
        """
        take an input image and return
            target_conv_output: the activation map of the selected target layer
            logit: the ourput tensor from the classifier
        """
        logit = self.model(x)
        target_conv_output = self.get_activations(x)
        return target_conv_output, logit

    def get_gradients(self):
        """
        after calling backwar() method, get the gradients
        """
        #after the mean calculation, tensor shape = [num_of_filters]
        return self.hook_g.squeeze(0).mean(axis=(1, 2))

    def generate_heatmap(self, transformed_img):
        """
        generate_heatmap: the important method that responsible of generating the
                            class activation map
        args:
            transformed_img: input image been transformed and pre-processed
        returns:
            the heatmap of the selected layer
        """
        conv_out, pred = self.forward_pass(transformed_img)
        conv_out.detach()

        class_id = pred.argmax(dim=1)
        class_id = class_id[0].data
        self.class_output = int(class_id)

        self.model.zero_grad()
        pred[:,class_id].backward(retain_graph=True)

        gradients = self.get_gradients()
        num_of_filters = gradients.shape[0]
        print(num_of_filters)
        for i in range(num_of_filters):
            #weighted sum-eq1
            conv_out[:, i, :, :] *= gradients[i]
        #the mean of the seighted sum of the gradients-eq2
        heatmap = torch.mean(conv_out, dim=1).squeeze().detach()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= torch.max(heatmap)
        return heatmap

    def visualize_heatmap(self,heatmap, img_path, alpha=0.5):
        """
        visualize_heatmap:
        args:
            heatmap: the heatmap at the same shape as the activation map
            img_path: input image path
        returns:
            overlap heatmap
        """
        img = cv2.imread(img_path)
        heatmap = cv2.resize(np.float64(heatmap), (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        self.save_heatmap(img_path, heatmap, img, alpha)
        #BGR for visualization
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        s_map_bgr = Image.blend(Image.fromarray(heatmap), Image.fromarray(img), alpha=alpha)
        return s_map_bgr

    def get_output_class(self):
        """return the class id (or) the argmax of the output tensor"""
        return self.class_output

    def save_heatmap(self, img_path,heatmap, img, alpha):
        '''
        save rgb heatmap
        '''
        #RGB for saving to disc
        s_map_rgb = Image.blend(Image.fromarray(heatmap), Image.fromarray(img), alpha=alpha)
        cv2.imwrite(img_path+'_torch_cam.jpg', np.array(s_map_rgb))
