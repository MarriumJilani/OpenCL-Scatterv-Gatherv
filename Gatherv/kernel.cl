__kernel void Gatherv(
    __global const int *sendbuf,
    __global const int *sendbuf2,
    __global int *rcvbuf,
    __global int *sendcountbuf,
    __global int *displacementbuf,
    __global int *localsum_buf,
    __global int *coordinating_process_buffer)
{
    int gid = get_global_id(0);
    int coord_process = coordinating_process_buffer[0];
    const int sendcount = sendcountbuf[gid];
    const int displacement = displacementbuf[gid];
    int is_coordinating_process = 0;
    if (gid == coord_process) 
	{
        is_coordinating_process = 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
   // if (is_coordinating_process) {}

    // Copy data from sendbuf to rcvbuf for the current process
	for (int i = displacement; i < displacement + sendcount; i++) {
		rcvbuf[i] = sendbuf[i]*sendbuf2[i];
	}
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    int local_dot = 0;
    for (int i = displacement; i < displacement + sendcount; i++)
	{
		local_dot += rcvbuf[i];
	}
    
    barrier(CLK_LOCAL_MEM_FENCE);
    localsum_buf[gid] = local_dot;
   // printf("LOCAL SUM for %d: %d\n", gid,localsum_buf[gid] );
    barrier(CLK_LOCAL_MEM_FENCE);

 
}
