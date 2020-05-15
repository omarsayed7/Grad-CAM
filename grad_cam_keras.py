import keras.backend as K
import numpy as np
import cv2
from PIL import Image
class Keras_GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.class_output = None

    def get_activations(self):
        return self.model.get_layer(self.target_layer)

    def get_predict(self, img):
        return self.model.predict(img)

    def get_gradients(self,output_tensor, activations):
        class_id = np.argmax(output_tensor[0])
        self.class_output = class_id
        class_output = self.model.output[:, class_id]
        #shape (None, 7, 7, 2048)
        gradients = K.gradients(class_output, activations.output)[0]
        #shape (2048,)
        pool_gradients = K.mean(gradients, axis=(0, 1, 2))
        return pool_gradients

    def generate_heatmap(self, tarnsformed_img):
        output = self.get_predict(tarnsformed_img)#placeholder
        activations = self.get_activations()#placeholder
        grads = self.get_gradients(output, activations)#placeholder

        num_of_filters = grads.shape[0]
        forward = K.function([self.model.input], [grads, activations.output[0]])#forward path
        grads, target_conv_layer = forward([tarnsformed_img]) # the actual tensors

        #time to compute equation-1 (multiply each activation map with corresponding pooled gradients)
        for i in range(num_of_filters):
            target_conv_layer[:,:,i] *= grads[i]

        #time to compute equation-2
        #        (get the mean of the weighted activation mapsto get the final heatmap then apply RELU)
        heatmap = np.mean(target_conv_layer,axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
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
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
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
        cv2.imwrite(img_path+'_cam.jpg', np.array(s_map_rgb))
