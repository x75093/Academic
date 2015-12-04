

// 3x3 Derivative of Gaussian filter
__kernel void
gaussian_first_axis(__global __read_only float *in_values,
           __global __write_only float *out_zero,
           __global __write_only float *out_first,
           __local float *buffer,
           int w, int h,
           int buf_w, int buf_h,
           const int halo, 
           __global __read_only float *direct_der,
           __global __read_only float *zero_kernel)
{

    //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0); // values for the columns
    const int y = get_global_id(1); // values for the rows

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // these will be values for the columns
    const int ly = get_local_id(1); // these will be values for the rows

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo; // buffer position
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 2


    //////////////////////loop to build Buffer///////////////////////

    //Iterate down each colum, using a row iterator
        // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
	    for (int row = 0; row < buf_h; row++) 
	   	{

	      int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
	      int max_y = buf_corner_y + row; //stepping by rows adjust y
	      int new_h = h - 1; // height index
	      int new_w = w - 1; // width index

	      // Load the values into the buffer
	      // This is a read from global memory global read
	      // Each thread is loading values into the buffer down columns
	      buffer[row * buf_w + idx_1D] = in_values[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
	    }
	}

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        float xneighbor_0 = buffer[(buf_y - 4) * buf_w + (buf_x - 0)]; 
        float xneighbor_1 = buffer[(buf_y - 3) * buf_w + (buf_x - 0)];
        float xneighbor_2 = buffer[(buf_y - 2) * buf_w + (buf_x - 0)];
        float xneighbor_3 = buffer[(buf_y - 1) * buf_w + (buf_x - 0)];
        float xneighbor_5 = buffer[(buf_y + 1) * buf_w + (buf_x + 0)]; 
        float xneighbor_6 = buffer[(buf_y + 2) * buf_w + (buf_x + 0)];
        float xneighbor_7 = buffer[(buf_y + 3) * buf_w + (buf_x + 0)];
        float xneighbor_8 = buffer[(buf_y + 4) * buf_w + (buf_x + 0)];
        float xpixel = buffer[(buf_y + 0) * buf_w + (buf_x + 0)];
        //use a for loop to multiply by
        float first = xneighbor_0 * direct_der[0] + xneighbor_1 * direct_der[1] + xneighbor_2 * 
        direct_der[2] + xneighbor_3 * direct_der[3] + xneighbor_5 * direct_der[5] + xneighbor_6 * direct_der[6] + 
        xneighbor_7 * direct_der[7] + xneighbor_8 * direct_der[8] + xpixel * direct_der[4];
        // + xpixel * direct_der[4]

        float zero = xneighbor_0 * zero_kernel[0] + xneighbor_1 * zero_kernel[1] + xneighbor_2 * 
        zero_kernel[2] + xneighbor_3 * zero_kernel[3] + xneighbor_5 * zero_kernel[5] + xneighbor_6 * zero_kernel[6] + 
        xneighbor_7 * zero_kernel[7] + xneighbor_8 * zero_kernel[8] + xpixel * zero_kernel[4];
        //


      	out_zero[y * w + x] = zero;
      	out_first[y * w + x] = first;

      }

}






__kernel void
gaussian_second_axis(__global __read_only float *in_zero,
           __global __read_only float *in_first,
           __global __write_only float *out_Wxx,
           __global __write_only float *out_Wyy,
           __global __write_only float *out_Wxy,
           __local float *buffer_order0,
           __local float *buffer_order1,
           int w, int h,
           int buf_w, int buf_h,
           const int halo, 
           __global __read_only float *direct_der,
           __global __read_only float *zero_kernel)
{

    //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0); // values for the columns
    const int y = get_global_id(1); // values for the rows

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // these will be values for the columns
    const int ly = get_local_id(1); // these will be values for the rows

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo; // buffer position
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 2


    //////////////////////loop to build Buffer///////////////////////

    //Iterate down each colum, using a row iterator
        // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
      for (int row = 0; row < buf_h; row++) 
      {

        int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
        int max_y = buf_corner_y + row; //stepping by rows adjust y
        int new_h = h - 1; // height index
        int new_w = w - 1; // width index

        // Load the values into the buffer
        // This is a read from global memory global read
        // Each thread is loading values into the buffer down columns
        buffer_order0[row * buf_w + idx_1D] = in_zero[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        buffer_order1[row * buf_w + idx_1D] = in_first[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
      }
  }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    //////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {

        float yneighbor_0_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 4)]; 
        float yneighbor_1_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 3)];
        float yneighbor_2_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 2)];
        float yneighbor_3_order0 = buffer_order0[(buf_y - 0) * buf_w + (buf_x - 1)];
        float yneighbor_5_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        float yneighbor_6_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 2)];
        float yneighbor_7_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 3)];
        float yneighbor_8_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 4)];
        float ypixel_order0 = buffer_order0[(buf_y + 0) * buf_w + (buf_x + 0)];

        float yneighbor_0_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 4)]; 
        float yneighbor_1_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 3)];
        float yneighbor_2_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 2)];
        float yneighbor_3_order1 = buffer_order1[(buf_y - 0) * buf_w + (buf_x - 1)];
        float yneighbor_5_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        float yneighbor_6_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 2)];
        float yneighbor_7_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 3)];
        float yneighbor_8_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 4)];
        float ypixel_order1 = buffer_order1[(buf_y + 0) * buf_w + (buf_x + 0)];

        //use a for loop to multiply by
        float Ix = yneighbor_0_order0 * direct_der[0] + yneighbor_1_order0 * direct_der[1] + yneighbor_2_order0 * 
        direct_der[2] + yneighbor_3_order0 * direct_der[3] + ypixel_order0 * direct_der[4] + yneighbor_5_order0 * direct_der[5] + yneighbor_6_order0 * direct_der[6] + 
        yneighbor_7_order0 * direct_der[7] + yneighbor_8_order0 * direct_der[8];

        float Iy = yneighbor_0_order1 * zero_kernel[0] + yneighbor_1_order1 * zero_kernel[1] + yneighbor_2_order1 * 
        zero_kernel[2] + yneighbor_3_order1 * zero_kernel[3] + ypixel_order1 * zero_kernel[4] + yneighbor_5_order1 * zero_kernel[5] + yneighbor_6_order1 * zero_kernel[6] + 
        yneighbor_7_order1 * zero_kernel[7] + yneighbor_8_order1 * zero_kernel[8];


        out_Wxx[y * w + x] = Ix; //Ix * Ix;
        out_Wyy[y * w + x] = Iy; //Iy * Iy;
        out_Wxy[y * w + x] = Ix * Iy;

      }

}










__kernel void 
gaussian_filter(__global __read_only float *in_Wxx,
      __global __read_only float *in_Wyy,
      __global __read_only float *in_Wxy, 
      __global __write_only float *out_W,
      __local float *buffer_Wxx,
      __local float *buffer_Wyy,
      __local float *buffer_Wxy,
      const int halo,
      int w, int h,
      int buf_w, int buf_h,
      __global __read_only float *filter)
{

   //////////////////////Define Variables///////////////////////

    // halo is the additional number of cells in one direction

    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Local position relative to (0, 0) in workgroup
    const int lx = get_local_id(0); // with in workgroup, so less than buffer
    const int ly = get_local_id(1);

    // coordinates of the upper left corner of the buffer in image (global)
    // space, including halo
    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    // coordinates of our pixel in the local buffer
    // this shifts the buffer reference to the middle of the buffer
    // where these pixels actualy exist in the buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    // 1D index of thread within our work-group
    const int idx_1D = ly * get_local_size(0) + lx; //get_local_size = 8

    //////////////////////loop to build Buffer///////////////////////

    // Load the relevant labels to a local buffer with a halo 
    if (idx_1D < buf_w) 
    {
        //Iterate down each colum, using a row iterator
        for (int row = 0; row < buf_h; row++) 
       {

          int max_x = buf_corner_x + idx_1D; // this is column index, add idx_1D
          int max_y = buf_corner_y + row; //stepping by rows adjust y
          int new_h = h - 1; // height index
          int new_w = w - 1; // width index

          // Load the values into the buffer
          // This is a read from global memory global read
          // Each thread is loading values into the buffer down columns
          buffer_Wxx[row * buf_w + idx_1D] = in_Wxx[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wyy[row * buf_w + idx_1D] = in_Wyy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
          buffer_Wxy[row * buf_w + idx_1D] = in_Wxy[min(max(0, max_y), new_h) * w + min(max(0, max_x), new_w)];
        }

    }

    // Make sure all threads reach the next part after
    // the local buffer is loaded
    barrier(CLK_LOCAL_MEM_FENCE);

//////////////////////coniditional statement to smooth///////////////////////

    // Conditional with in bounds of the entire image
    if (x < w && y < h)
      {
        // Create new index, 'idx,' for reference within function call
        // to median9
        //int idx = (lx + 1) + (2 * halo + get_local_size(0)) * (ly + 1); 

        float neighbor_0_Wxx = buffer_Wxx[(buf_y - 1) * buf_w + (buf_x - 1)]; 
        float neighbor_1_Wxx = buffer_Wxx[(buf_y - 1) * buf_w + (buf_x + 0)];
        float neighbor_2_Wxx = buffer_Wxx[(buf_y - 1) * buf_w + (buf_x + 1)];
        float neighbor_3_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x - 1)];
        float neighbor_5_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        float neighbor_6_Wxx = buffer_Wxx[(buf_y + 1) * buf_w + (buf_x - 1)];
        float neighbor_7_Wxx = buffer_Wxx[(buf_y + 1) * buf_w + (buf_x + 0)];
        float neighbor_8_Wxx = buffer_Wxx[(buf_y + 1) * buf_w + (buf_x + 1)];
        float pixel_Wxx = buffer_Wxx[(buf_y + 0) * buf_w + (buf_x + 0)];
        //use a nested for loop
        float Wxx_filter = pixel_Wxx * (filter[4]) + neighbor_0_Wxx * (filter[0]) + neighbor_1_Wxx * (filter[1]) +
         neighbor_2_Wxx * (filter[2]) + neighbor_3_Wxx * (filter[3]) + neighbor_5_Wxx * (filter[5]) + 
         neighbor_6_Wxx * (filter[6]) + neighbor_7_Wxx * (filter[7]) + neighbor_8_Wxx * (filter[8]);
 
        //out_Wxx[y * w + x] = Wxx_filter;     

        float neighbor_0_Wyy = buffer_Wyy[(buf_y - 1) * buf_w + (buf_x - 1)]; 
        float neighbor_1_Wyy = buffer_Wyy[(buf_y - 1) * buf_w + (buf_x + 0)];
        float neighbor_2_Wyy = buffer_Wyy[(buf_y - 1) * buf_w + (buf_x + 1)];
        float neighbor_3_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x - 1)];
        float neighbor_5_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        float neighbor_6_Wyy = buffer_Wyy[(buf_y + 1) * buf_w + (buf_x - 1)];
        float neighbor_7_Wyy = buffer_Wyy[(buf_y + 1) * buf_w + (buf_x + 0)];
        float neighbor_8_Wyy = buffer_Wyy[(buf_y + 1) * buf_w + (buf_x + 1)];
        float pixel_Wyy = buffer_Wyy[(buf_y + 0) * buf_w + (buf_x + 0)];

        float Wyy_filter = pixel_Wyy * (filter[4]) + neighbor_0_Wyy * (filter[0]) + neighbor_1_Wyy * (filter[1]) +
         neighbor_2_Wyy * (filter[2]) + neighbor_3_Wyy * (filter[3]) + neighbor_5_Wyy * (filter[5]) + 
         neighbor_6_Wyy * (filter[6]) + neighbor_7_Wyy * (filter[7]) + neighbor_8_Wyy * (filter[8]);
  

        float neighbor_0_Wxy = buffer_Wxy[(buf_y - 1) * buf_w + (buf_x - 1)]; 
        float neighbor_1_Wxy = buffer_Wxy[(buf_y - 1) * buf_w + (buf_x + 0)];
        float neighbor_2_Wxy = buffer_Wxy[(buf_y - 1) * buf_w + (buf_x + 1)];
        float neighbor_3_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x - 1)];
        float neighbor_5_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 1)]; 
        float neighbor_6_Wxy = buffer_Wxy[(buf_y + 1) * buf_w + (buf_x - 1)];
        float neighbor_7_Wxy = buffer_Wxy[(buf_y + 1) * buf_w + (buf_x + 0)];
        float neighbor_8_Wxy = buffer_Wxy[(buf_y + 1) * buf_w + (buf_x + 1)];
        float pixel_Wxy = buffer_Wxy[(buf_y + 0) * buf_w + (buf_x + 0)];

        float Wxy_filter = pixel_Wxy * (filter[4]) + neighbor_0_Wxy * (filter[0]) + neighbor_1_Wxy * (filter[1]) +
         neighbor_2_Wxy * (filter[2]) + neighbor_3_Wxy * (filter[3]) + neighbor_5_Wxy * (filter[5]) + 
         neighbor_6_Wxy * (filter[6]) + neighbor_7_Wxy * (filter[7]) + neighbor_8_Wxy * (filter[8]);
 

        float Wdet = Wxx_filter * Wyy_filter - Wxy_filter * Wxy_filter;
        float Wtr = Wxx_filter + Wyy_filter;
        float ans = Wdet / Wtr;

        out_W[y * w + x] = ans; 

      } 
  

}










