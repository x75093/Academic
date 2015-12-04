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


def generate_weights(sigma, order = 1):
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
    if order == 1:
        weights[lw] = 0.0
        for ii in range(1, lw + 1):
            x = float(ii)
            tmp = -x / sd * weights[lw + ii]
            weights[lw + ii] = -tmp
            weights[lw - ii] = tmp

    return weights

def get_harris_points(harrisim, min_dist=10, threshold=.1): 
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
    print len(filtered_coords)

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
    program = cl.Program(context, open('quad_filter_1.cl').read()).build(options='')

   

    host_image = np.array(Image.open('golden_gate.jpg').convert('L')).astype(np.float32)[::1, ::1].copy()
    host_image64 = np.array(Image.open('golden_gate.jpg').convert('L')).astype(np.float64)[::1, ::1].copy()

    #host_image = np.array([np.arange(2)]*2).astype(np.float32)[::1, ::1].copy()
    start = time.time()
 
    sigma = 1 #only works for sigma of 1
    filter_kernel_x = np.asarray(generate_weights(sigma)).astype(np.float32)
    filter_kernel_x64 = np.asarray(generate_weights(sigma))
    filter_kernel_zero = np.asarray(generate_weights(sigma, order = 0)).astype(np.float32)
    weight_length = len(filter_kernel_x) #should be 9 with sigma = 1
    window = (weight_length - 1) #window is 8
    halo = np.int32(window / 2.)



    host_image_filtered = np.zeros_like(host_image)

    gpu_image_in = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    #gpu_image_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    zero_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    first_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    derivative_kernel_x = cl.Buffer(context, cl.mem_flags.READ_WRITE, filter_kernel_x.size * 4)
    zero_kernel = cl.Buffer(context, cl.mem_flags.READ_WRITE, filter_kernel_zero.size * 4)

    # Intermediate storage area, between Derivative of Gaussian and Gaussian Filter
    host_image_Wxx = np.zeros_like(host_image)
    host_image_Wyy = np.zeros_like(host_image)
    host_image_Wxy = np.zeros_like(host_image)

    Harris_Matrix = np.zeros_like(host_image)

    # Intermediate storage area, between Derivative of Gaussian and Gaussian Filter
    host_image_temp = np.zeros_like(host_image)

    local_size = (int(halo), int(halo)) # 2D local_size
    global_size = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size)]) # shape

    width = np.int32(host_image.shape[1])
    height = np.int32(host_image.shape[0])

    local_memory = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    #local_memory = cl.LocalMemory(4 * ((local_size[0] * local_size[1]) + window))
	

    buf_width = np.int32(local_size[0] + window)
    buf_height = np.int32(local_size[1] + window)

    #Redefine work group size and global size for next kernel
    local_size_2 = (8,8) # 2D local_size
    global_size_2 = tuple([round_up(g, l) for g, l in zip(host_image.shape[::-1], local_size_2)]) # shape
	
    buf_height_2 = np.int32(local_size_2[1] + 2)
    halo_2 = np.int32(1)
    buf_width_2 = np.int32(local_size_2[0] + 2)

    cl.enqueue_copy(queue, gpu_image_in, host_image, is_blocking=False)
    cl.enqueue_copy(queue, derivative_kernel_x, filter_kernel_x, is_blocking=False)
    cl.enqueue_copy(queue, zero_kernel, filter_kernel_zero, is_blocking=False)

    #Execute Derivative of Gaussian Function
    event_dertivative = program.gaussian_first_axis(queue, global_size, local_size,
                        gpu_image_in, 
                        zero_derivative_out, 
                        first_derivative_out, 
                        local_memory, width, 
                        height, buf_width, buf_height, halo, 
                        derivative_kernel_x, zero_kernel)

    event_dertivative.wait()

    cl.enqueue_copy(queue, host_image_Wyy, first_derivative_out, is_blocking=True)


    local_memory_axis2_1 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    local_memory_axis2_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    gpu_image_Wxx_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_Wyy_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_Wxy_derivative_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

    #Execute Derivative of Gaussian Function
    event_dertivative = program.gaussian_second_axis(

                        queue, global_size, local_size, 
                        zero_derivative_out, 
                        first_derivative_out, 
                        gpu_image_Wxx_derivative_out, 
                        gpu_image_Wyy_derivative_out, 
                        gpu_image_Wxy_derivative_out,
                        local_memory_axis2_1, local_memory_axis2_2,
                        width, 
                        height, buf_width, buf_height, halo, 
                        derivative_kernel_x, zero_kernel

                        )

    event_dertivative.wait()
 


    # Get Data out of Derivative of Gaussian GPU function
    #cl.enqueue_copy(queue, host_image_Wxx, gpu_image_Wxx_derivative_out, is_blocking=True)
    #cl.enqueue_copy(queue, host_image_Wyy, gpu_image_Wyy_derivative_out, is_blocking=True)
    #cl.enqueue_copy(queue, host_image_Wxy, gpu_image_Wxy_derivative_out, is_blocking=True)


    #Create new data structure for transer between kernels, into second kernel
    gpu_image_Wxx_filter_in = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_Wyy_filter_in = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)
    gpu_image_Wxy_filter_in = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

    # Requeue Data from Dervative of Guassian for Gaussian Filter
    # cl.enqueue_copy(queue, gpu_image_Wxx_filter_in, host_image_Wxx, is_blocking=False)
    # cl.enqueue_copy(queue, gpu_image_Wyy_filter_in, host_image_Wxx, is_blocking=False)
    # cl.enqueue_copy(queue, gpu_image_Wxy_filter_in, host_image_Wxx, is_blocking=False)

    #load in the local memory buffer allocation for all the compents fo the harris matrix
    #local_memory_derivative = cl.LocalMemory(4 * (local_size_2[0] + 2) * (local_size_2[1] + 2))
    local_memory_filter_Wxx = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    local_memory_filter_Wyy = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    local_memory_filter_Wxy = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))

    gpu_image_filter_out = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_image.size * 4)

   # Execute gaussian filter on all component matrices and calculate Harris Matrix
    event_filter = program.filter_first_axis_second_pass(

                        queue, global_size, local_size, 
                        gpu_image_Wxx_derivative_out, 
                        gpu_image_Wyy_derivative_out, 
                        gpu_image_Wxy_derivative_out,
                        gpu_image_Wxx_filter_in, 
                        gpu_image_Wyy_filter_in, 
                        gpu_image_Wxy_filter_in, 
                        local_memory_filter_Wxx, 
                        local_memory_filter_Wyy, 
                        local_memory_filter_Wxy,
                        halo, width, height, buf_width, buf_height, 
                        zero_kernel

                        )

    event_filter.wait()

    cl.enqueue_copy(queue, host_image_Wxy, gpu_image_Wxx_filter_in, is_blocking=True)
    local_memory_filter_Wxx_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    local_memory_filter_Wyy_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))
    local_memory_filter_Wxy_2 = cl.LocalMemory(4 * ((local_size[0] + (halo * 2)) * (local_size[1] + (halo * 2))))


    event_filter = program.filter_second_axis_second_pass(

                        queue, global_size, local_size, 
                        gpu_image_Wxx_filter_in, 
                        gpu_image_Wyy_filter_in, 
                        gpu_image_Wxy_filter_in, 
                        gpu_image_filter_out,
                        local_memory_filter_Wxx_2, 
                        local_memory_filter_Wyy_2, 
                        local_memory_filter_Wxy_2,
                        halo, width, height, buf_width, buf_height, 
                        zero_kernel

                        )

    event_filter.wait()




    cl.enqueue_copy(queue, Harris_Matrix, gpu_image_filter_out, is_blocking=False)
    # start1 = time.time()
    points = get_harris_points(Harris_Matrix)
    # end = time.time()
    # t = end - start
    # t1 = end - start1
    
    plot_harris_points(host_image, points)
    
    pt_x = 0
    pt_y = 200


    print 'compare_cl y first', host_image_Wyy[pt_x , pt_y]
    cor_x1 = filters.correlate1d(host_image64,filter_kernel_x64,0, mode = 'nearest')
    print 'compare scipy y first', cor_x1[pt_x , pt_y]


    # print 'compare_cl x', host_image_Wxx[pt_x , pt_y:pt_y + 4]
    imx = np.zeros(host_image64.shape)
    filters.gaussian_filter(host_image64, (sigma,sigma), (0,1), imx, mode = 'nearest')
    print 'scipy gaussian x', imx[pt_x , pt_y : pt_y + 4] * imx[pt_x , pt_y : pt_y + 4]


    print 'Harris Matrix', Harris_Matrix[pt_x, pt_y]
    cor_third_event = filters.correlate1d(imx*imx,filter_kernel_zero ,0, mode = 'nearest')
    print 'thirds pass cl', host_image_Wxy[pt_x, pt_y]
    print 'third pass scipy', cor_third_event[pt_x, pt_y]










