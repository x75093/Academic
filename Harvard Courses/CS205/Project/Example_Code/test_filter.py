from __future__ import division
import pyopencl as cl
import numpy as np
import Image
from PIL import Image
from skimage import color
import pylab
from scipy.ndimage import filters
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy import linalg
import time

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r


def generate_weights(sigma):
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    truncate = 4.0
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    weights[lw] = 0.0
    for ii in range(1, lw + 1):
        x = float(ii)
        tmp = -x / sd * weights[lw + ii]
        weights[lw + ii] = -tmp
        weights[lw - ii] = tmp

    return weights

def get_harris_points(harrisim, min_dist=15, threshold=.15): 
    """ Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary. """
    
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1
    
    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords] # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = [] 
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i]) 
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                              (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0 
    
    return filtered_coords

def plot_harris_points(image,filtered_coords):
    """ Plots corners found in image. """
    plt.figure(figsize=[12,8])
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*') 
    plt.axis('off')
    plt.show()

def check_dim(im):
    print "Shape of image:", im.shape
    if len(im.shape) > 2:
    # Take average of each pixel did not work
    #im_avg = (im[:, :, 0] + im[:, :, 1] + im[:, :, 2])/3
    
        # So I just returned one part of image
        return im[:, :, 1]
    
    else:
        return im

if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'
    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name
    program = cl.Program(context, open('test_filter.cl').read()).build(options='')

    
    host_image = np.array(Image.open('golden_gate.jpg').convert('L')).astype(np.float32)[::1, ::1].copy()
    start = time.time()
    sigma = 1 #only works for sigma of 1
    filter_kernel_x = np.asarray(generate_weights(sigma)).astype(np.float32)
    weight_length = len(filter_kernel_x) #should be 9 with sigma = 1
    window = (weight_length - 1) #window is 8
    window_length_in = np.int32(weight_length)
    halo = np.int32(window / 2.)
    window_range = np.asarray(range(-halo, halo + 1)).astype(np.float32)



    host_image_filtered = np.zeros_like(host_image)

    gpu_image_in = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    window_range_in = cl.Buffer(context, cl.mem_flags.READ_WRITE, window_range.size * 4)
    #gpu_image_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_Wxx_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_Wyy_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_Wxy_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    derivative_kernel_x = cl.Buffer(context, cl.mem_flags.READ_WRITE, filter_kernel_x.size * 4)

    # Intermediate storage area, between Derivative of Gaussian and Gaussian Filter
    # host_image_Wxx = np.zeros_like(host_image)
    # host_image_Wyy = np.zeros_like(host_image)
    # host_image_Wxy = np.zeros_like(host_image)

    Harris_Matrix = np.zeros_like(host_image)

    # Intermediate storage area, between Derivative of Gaussian and Gaussian Filter
    host_image_temp = np.zeros_like(host_image)

    local_size = (int(halo), int(halo)) # 2D local_size
    global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)]) # shape

    width = np.int32(host_image.shape[1])
    height = np.int32(host_image.shape[0])

    local_memory = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    local_memory_der = cl.LocalMemory(4 * filter_kernel_x.size)

    buf_width = np.int32(local_size[0] + window)
    buf_height = np.int32(local_size[1] + window)

    #Redefine work group size and global size for next kernel
    local_size_2 = (8,8) # 2D local_size
    global_size_2 = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size_2)]) # shape
    
    buf_height_2 = np.int32(local_size_2[1] + 2)
    halo_2 = np.int32(1)
    buf_width_2 = np.int32(local_size_2[0] + 2)

    cl.enqueue_copy(queue, window_range_in, window_range, is_blocking=False)
    cl.enqueue_copy(queue, gpu_image_in, host_image, is_blocking=False)
    cl.enqueue_copy(queue, derivative_kernel_x, filter_kernel_x, is_blocking=False)

    #Execute Derivative of Gaussian Function
    event_dertivative = program.derivative_of_gaussian(queue, global_size, local_size,
                        gpu_image_in, gpu_image_Wxx_derivative_out, 
                        gpu_image_Wyy_derivative_out, gpu_image_Wxy_derivative_out,
                        local_memory, width, 
                        height, buf_width, buf_height, halo, 
                        derivative_kernel_x, local_memory_der,
                        window_range_in, window_length_in)

    event_dertivative.wait()
 


    # Get Data out of Derivative of Gaussian GPU function
    # cl.enqueue_copy(queue, host_image_Wxx, gpu_image_Wxx_derivative_out, is_blocking=True)
    # cl.enqueue_copy(queue, host_image_Wyy, gpu_image_Wyy_derivative_out, is_blocking=True)
    # cl.enqueue_copy(queue, host_image_Wxy, gpu_image_Wxy_derivative_out, is_blocking=True)


    #load in the local memory buffer allocation for all the compents fo the harris matrix
    local_memory_derivative = cl.LocalMemory(4 * (local_size_2[0] + 2) * (local_size_2[1] + 2))
    local_memory_filter_Wxx = cl.LocalMemory(4 * (local_size_2[0] + 2) * (local_size_2[1] + 2))
    local_memory_filter_Wyy = cl.LocalMemory(4 * (local_size_2[0] + 2) * (local_size_2[1] + 2))
    local_memory_filter_Wxy = cl.LocalMemory(4 * (local_size_2[0] + 2) * (local_size_2[1] + 2))

    gpu_image_filter_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

    gaussian_kernel = np.array([[1./16., 1./8., 1./16.],[1./8., 1./4., 1./8.],[1./16., 1./8., 1./16.]]).astype(np.float32)
    local_memory_gaus = cl.LocalMemory(4 * gaussian_kernel.size)
    # Create the space for the Guassian Filter kernel 
    filter_kernel = cl.Buffer(context, cl.mem_flags.READ_WRITE, gaussian_kernel.size * 4)
    cl.enqueue_copy(queue, filter_kernel, gaussian_kernel, is_blocking=False)


    # Execute gaussian filter on all component matrices and calculate Harris Matrix
    event_filter = program.gaussian_filter(queue, global_size_2, local_size_2, gpu_image_Wxx_derivative_out, 
        gpu_image_Wyy_derivative_out, gpu_image_Wxy_derivative_out, 
        gpu_image_filter_out,
        local_memory_filter_Wxx, local_memory_filter_Wyy, local_memory_filter_Wxy,
        halo_2, width, height, buf_width_2, buf_height_2, filter_kernel, local_memory_gaus)

    event_filter.wait()

    cl.enqueue_copy(queue, Harris_Matrix, gpu_image_filter_out, is_blocking=False)
    start1 = time.time()
    points = get_harris_points(Harris_Matrix)
    end = time.time()
    t = end - start
    t1 = end - start1
    
    #plot_harris_points(host_image, points)
    #print host_image_Wxx[0,:10]
    print 'total', t
    print 'points', len(points)










