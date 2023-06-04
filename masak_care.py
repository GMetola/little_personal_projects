import datetime
import glob
import os
import time
import threading

import cv2
import discord
import numpy as np
# from matplotlib import pyplot as plt


discord_active = True
with open(r'C:\Users\metol\Documents\Discord Masak token.txt', 'r') as f:
    lines = f.readlines()
bot_token, channel_id = lines[0], int(lines[1])


if discord_active:
    # Create a new instance of the Discord client with intents
    intents = discord.Intents.default()
    intents.dm_messages = True
    intents.dm_typing = True
    intents.guild_messages = True
    intents.guild_typing = True
    intents.guilds = True
    intents.messages = True
    intents.typing = True
    intents.message_content = True
    client = discord.Client(intents=intents)
    print(f'We have logged in as {client.user}')


class Camera:
    '''
        Class that includes everything related to the camera.
    '''

    def __init__ (  self,
                    exposure=None,
                    gain=None,
                    width=None,
                    height=None,
                    framerate=2,
                    path='C:/IMAGES/cat',
                    format_img='',
                    visualize=True,
                    debug=False):
        # Definition
        self.input_path = path
        self.output_path = 'C:/IMAGES/cat'
        self.format = format_img
        self.desired_exposure = exposure
        self.desired_gain = gain
        self.debug = debug
        self.window_name = 'Vigilando al gato'
        self.width = width
        self.height = height

        # Camera matrices
        self.intrinsic = np.eye(3)
        self.mrotation = np.zeros((3,3))
        self.mtraslation = np.zeros((3))
        self.coef_distortion = np.zeros(5)
        self.optical_center = np.zeros((3))
        self.frame = np.zeros((3))

        # State
        self.counter = 0
        self.not_finished = 1
        self.recording = False
        self.frame_number = 0  # Starting image

        self.camera_type = 'VIRTUAL_CAMERA'
        self.input_path = os.path.join(path,'camera/')
        self.masks_path = os.path.join(path,'masks/')
        self.files = glob.glob(self.input_path + '*' + self.format)
        self.visualize = visualize
        if self.frame_number != 0:
            print("Virtual camera. Starting image number: %s", int(self.frame_number))

    def calibration (self):
        '''
            In this method the camera is calibrated individually,
            extracting its intrinsic and extrinsic parameters.
        '''
        print('Camera calibration.')

        image_list, image_size = self.load_images()
        print('Loaded '+ str(len(image_list))
            + ' images from directory ' + self.input_path)

        points_corners_2D, points__2D_refined = self.__get_points_2D(image_list)

        points_corners_3D = self.__get_chessboard_points_3D()

        # Prepare input variables for calibration.
        points_2D_valid = [corner for exito,corner in points_corners_2D if exito]
        points_2D = np.array([corner[:,0,:] for corner in points_2D_valid],
                    dtype=np.float32)
        points_3D = np.array([points_corners_3D] * len(points_2D_valid),
                dtype=np.float32)
        print('puntos 3D: ' + str(points_3D.shape))
        print('points 2D: ' + str(points_2D.shape))

        # Calibracion:
        rms, self.intrinsic, self.coef_distortion, rotation, tras = cv2.calibrateCamera(
            points_3D, points_2D, image_size, self.coef_distortion,
            cv2.CALIB_FIX_ASPECT_RATIO)

        print('RMS re-projection error: ' + str(rms))
        print('Intrinsic: ')
        print(self.intrinsic)
        print('Distortion coefficients: ')
        print(self.coef_distortion)

        # Extract the pose and the optical center of the camera in 3D.
        # Find the rotation and translation vectors.
        retval, self.mrotation, self.mtraslation = cv2.solvePnP(points_corners_3D,
        points__2D_refined, self.intrinsic, self.coef_distortion)
        #optical_center = -matriz_rotation.T * t
        rotation_matrix, _ = cv2.Rodrigues(self.mrotation)
        rotation_matrix= rotation_matrix.transpose()
        self.optical_center = np.dot(-rotation_matrix,self.mtraslation)

        print('Rotation matrix: ')
        print(self.mrotation)
        print('Translation matrix: ')
        print(self.mtraslation)
        print('Optical Center: ')
        print(self.optical_center)

        return self.intrinsic, self.mrotation, self.mtraslation

    def stop(self):
        self.not_finished = 0

    def close(self):
        '''
            Free resources
        '''
        print("Close camera")

    def get_video(self, color_space='rgb', save_frames=False):
        if self.visualize:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        patience = 50

        # Open the camera
        for camera_index in range(10):
            cap = cv2.VideoCapture(camera_index)  # Use 0 as the argument for the default camera
            if not cap.isOpened():
                print("Failed to open camera", camera_index)
            else:
                print("Camera opened")
                break

        if not self.width is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Read and display video frames
        while True:
            # Read a frame from the camera
            ret, self.frame = cap.read()

            # Check if the frame was successfully read
            if not ret:
                print("Failed to read frame from camera")
                patience -= 1
                if patience <= 0:
                    break
            if color_space == 'hsv':
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            yield self.frame
            # Display the frame
            if self.visualize:
                cv2.imshow(self.window_name, self.frame)
                # Exit loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if save_frames:
                now = datetime.datetime.now()
                name = 'camera_'+ '/' + now.strftime('%y%m%d_%H%M%S') + '_' + str(self.counter).zfill(2) + '.bmp'
                out_filepath = os.path.join(self.output_path, name)


        # Release the camera and close the display window
        cap.release()
        cv2.destroyAllWindows()

    def get_image(self, color_space='rgb', visualize=False,exposure=-1,gain=-1) -> np.array:
        if visualize:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Open the camera
        for camera_index in range(10):
            cap = cv2.VideoCapture(camera_index)  # Use 0 as the argument for the default camera
            if not cap.isOpened():
                print("Failed to open camera", camera_index)
            else:
                print("Camera opened")
                break

        if not self.width is None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Read and display video frames
        patience = 20
        while patience > 0:
            # Read a frame from the camera
            ret, self.frame = cap.read()

            # Check if the frame was successfully read
            if not ret:
                print("Failed to read frame from camera")
                patience -= 1
                continue
            else:  # Got the frame
                patience = 0

            if color_space == 'hsv':
                self.frame = cv2.cvtColor(self.frame, cv2.color_bgr2hsv)

            # Display the frame
            if visualize:
                cv2.imshow(self.window_name, self.frame)
                # Exit loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release the camera and close the display window
        cap.release()
        if visualize:
            cv2.destroyAllWindows()

        return self.frame

    def record_video(self,record_framerate):
        self.counter = 1
        start_time = time.perf_counter()
        img = self.get_image()
        frame_width = img.shape[1]
        frame_height = img.shape[0]
        now = datetime.datetime.now()
        name = 'camera/'+ now.strftime('%y%m%d_%H%M%S') + '_' + str(self.counter).zfill(6)
        video_out = cv2.VideoWriter(name+'.mp4',cv2.VideoWriter_fourcc(*'X264'), record_framerate, (frame_width,frame_height))

        while self.not_finished:
            img = self.get_image()
            video_out.write(np.uint8(img))
            elapsed_time = time.perf_counter()-start_time
            target_time = self.counter/record_framerate
            difference = target_time - elapsed_time
            if difference > 0:
                time.sleep(difference)
            self.counter += 1

    def picture_mode(self):
        def take_picture(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                now = datetime.datetime.now()
                name = 'camera_'+ now.strftime('%y%m%d_%H%M%S') + '.bmp'
                out_filepath = os.path.join(self.output_path, name)
                cv2.imwrite(out_filepath,img)
                print("Image saved at", out_filepath)

        cv2.namedWindow('Camera ', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Camera ', take_picture)
        while 1:
            # Access the image data
            img = self.get_image()
            cv2.imshow('Camera', img)
            k = cv2.waitKey(1)
            if k == 27:             #Esc key to close
                break
        cv2.destroyAllWindows()
        print('Exiting picture mode')

    def trigger_mode(self):
        window_name = 'Camera' + " -- Click for image"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        def take_picture(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                img = self.get_image()
                cv2.imshow(window_name, img)

        cv2.setMouseCallback(window_name, take_picture)
        while 1:
            k = cv2.waitKey(1)
            if k == 27:             #Esc key to close
                break

        cv2.destroyAllWindows()
        print('Exiting picture mode')

    def load_images(self):
        '''
            Loads the images contained in a directory.
        '''
        file_list = glob.glob(self.input_path + '*' + self.format)
        image_list = [cv2.imread(file) for file in file_list]
        image_size = image_list[0].shape
        image_size = (image_size[0],image_size[1])

        return image_list, image_size


class ImageZone:
    """
    Mother class for Dispensers and Contents.
    Represents a zone in the image and its properties
    """
    project_path = 'C:/IMAGES/cat'
    base_img_path = os.path.join(project_path,'base.png')
    base_img = cv2.imread(base_img_path)

    # GIMP BASED <<- HSV as: ( 0-360 º, 0-100%, 0-100%)
    all_base_hsv_min = {'food': [5, 20, 20],
                        'comedero1':[200,5,15],
                        'comedero2':[190,60,50],
    }
    all_base_hsv_max = {'food':[35, 60, 40],
                        'comedero1':[250,40,100],
                        'comedero2':[220,100,70],
    }

    # OPENCV BASED <<- HSV as: ( 0-180, 0-255, 0-255)
    all_base_hsv_min = {'food': [2.5, 51.0, 51.0],
                        'comedero1': [100.0, 12.75, 38.25],
                        'comedero2': [95.0, 153.0, 128]}

    all_base_hsv_max = {'food': [17.5, 190.0, 130.0],
                        'comedero1': [125.0, 102.0, 255],
                        'comedero2': [110.0, 255, 178.5]}

    current_hsv = None

    def __init__(self,name) -> None:
        self.name = name
        self.mask_name = 'mask_' + name + '.png'

        self.mask_path = os.path.join(self.project_path, 'masks', self.mask_name)
        self.mask = cv2.imread(self.mask_path, 0)



class Content(ImageZone):
    """Food, water, bubbles and shit"""

    def __init__(self,name) -> None:
        super().__init__(name)
        self.food_hsv_min = self.all_base_hsv_min['food']
        self.food_hsv_max = self.all_base_hsv_max['food']


class Dispenser(ImageZone):
    """Each of the cat's dispensers"""

    def __init__(self,name) -> None:
        super().__init__(name)
        self.base_hsv_min = self.all_base_hsv_min[name]
        self.base_hsv_max = self.all_base_hsv_max[name]

class Comparer:
    """
    Gets current image and checks color and position
    """
    def __init__(self, camera) -> None:
        self.cam = camera
        self.current_img = cv2.cvtColor(self.cam.get_image(), cv2.COLOR_BGR2HSV)
        self.masked_img = None
        self.results = {}

    def hsv_base(self):
        pass

    def calibrate_hsv(self):
        """Compare current and base"""
        pass

    def get_average_hsv(self, mask, origin='image'):
        if origin != 'image':
            print("Not ready yet")
            return
        self.masked_img = cv2.bitwise_and(self.current_img, self.current_img, mask=mask)
        hue, sat, val = cv2.split(self.masked_img)
        print("")

    def find_content(self, content):
        self.masked_img = cv2.bitwise_and(self.current_img, self.current_img, mask=content.mask)
        low_thresh = self.masked_img > content.food_hsv_min
        high_thresh = self.masked_img < content.food_hsv_max
        food_mask = np.all(low_thresh & high_thresh, axis=2)
        content_percentage = int(np.sum(food_mask)/np.sum(content.mask==255)*100)
        print("Percentaje: ", content_percentage)
        self.results[content.name] = content_percentage
        return food_mask, content_percentage

    def live_comparison(self, arg):
        cv2.namedWindow("Comparer", cv2.WINDOW_NORMAL)
        video = self.cam.get_video(color_space='hsv')
        if discord_active:
            notifier("Comenzando comparacion")
        while True:
            # Update image
            self.current_img = next(video)

            # Get content calculation
            food_mask, content_percentage = self.find_content(arg)

            #Show result
            show_image = self.current_img.copy()
            show_image[food_mask] = (0,0,0)
            text = "Comedero azul: " + str(content_percentage)
            cv2.putText(show_image, text, (100, 100),cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),3)
            cv2.imshow("Comparer", show_image)

            # Exit loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow("Comparer")
                break

def notifier(message: str):
    @client.event
    async def on_message(message):
        if message.author == client.user:
            return

        if message.content.startswith('h'):
            await message.channel.send('Hello!')

    @client.event
    async def on_ready():
        channel = client.get_channel(channel_id)
        await channel.send(message)
        await client.close()

    client.run(bot_token)


if __name__ == '__main__':


    comedero1 = Dispenser('comedero1')
    comedero2 = Dispenser('comedero2')

    comedero1_contenido = Content('comedero1_contenido')
    comedero2_contenido = Content('comedero2_contenido')

    cam = Camera(   exposure = None,
                    gain = None,
                    width = 1920,
                    height = 1080,
                    path = 'C:/IMAGES/cat',
                    format_img = '.bmp',
                    visualize= False)
    print("Camera initialized")

    comparer = Comparer(cam)
    comparer.live_comparison(comedero2_contenido)
