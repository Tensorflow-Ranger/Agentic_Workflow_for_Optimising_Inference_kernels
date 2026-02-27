#define MAX_TAP 256

__kernel void FIR(__global float* output,
                  __global float* coeff,
                  __global float* input,
                  __global float* history,
                  uint num_tap)
{
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint lsize = get_local_size(0);

    __local float lcoeff[MAX_TAP];

    /* Load coefficient taps into LDS */
    for (uint i = lid; i < num_tap; i += lsize)
    {
        lcoeff[i] = coeff[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;

    uint limit = (gid < num_tap) ? gid : (num_tap - 1);

    /* Convolution using current input samples */
    for (uint i = 0; i <= limit; ++i)
    {
        sum += lcoeff[i] * input[gid - i];
    }

    /* Use history buffer for the remaining taps */
    for (uint i = limit + 1; i < num_tap; ++i)
    {
        uint idx = num_tap - (i - gid);
        sum += lcoeff[i] * history[idx];
    }

    output[gid] = sum;
}