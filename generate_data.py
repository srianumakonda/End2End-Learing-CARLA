import glob
import os
import sys
try:
    sys.path.append(glob.glob('../CARLA_0.9.11/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2

IM_WIDTH = 256
IM_HEIGHT = 128
j = [0]

def process_img(image, vehicle, j):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    data_val = vehicle.get_control() 
    print(data_val.steer, data_val.throttle, data_val.brake, data_val.reverse)

    with open('data.txt', 'a', encoding='utf-8') as f:
        f.write(f"{data_val.steer} {data_val.throttle} {data_val.brake} {data_val.reverse} \n")
        # f.close()

    cv2.imwrite(f"output/{j[0]}.jpg", i3)
    j[0]+=1
    cv2.imshow("", i3)
    cv2.waitKey(20)
    return i3/255.0


actor_list = []
try:

    with open('data.txt', 'a', encoding='utf-8') as f:
        f.write("steer throttle brake reverse \n")
        # f.close()

    client = carla.Client('localhost', 2000)
    client.set_timeout(200.0)
    # world = client.get_world()
    world = client.load_world('Town04')
    # print(client.get_available_maps())
    client.reload_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    vehicle.set_autopilot(True)

    actor_list.append(vehicle)

    blueprint = blueprint_library.find('sensor.camera.rgb')
    blueprint.set_attribute('image_size_x', f"{IM_WIDTH}")
    blueprint.set_attribute('image_size_y', f"{IM_HEIGHT}")
    # blueprint.set_attribute('fov', '110')
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
    actor_list.append(sensor)

    sensor.listen(lambda data: process_img(data, vehicle, j))
    time.sleep(60)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')
