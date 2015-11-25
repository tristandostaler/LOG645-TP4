/*__kernel void HeatTransfer()
{
	int id = get_global_id(0);
}*/

__kernel void HeatTransfer(__global const float* a, __global const int* w, __global const float* b, __global float* c)
{
	// get index into global data array
	float* a_c_p;
	float* c_a_p;
	int iGID = get_global_id(0);

	//if (iGID >= b[2] && iGID <= (b[2] * (b[3] - 1)) && iGID % b[2] != 0 && iGID % b[2] != b[2] - 1)
	int round = b[4];
	int modRes = b[4] % 2;
	
	if (modRes == 0)
	{
		a_c_p = *a;
		c_a_p = *c;
	}	
	else
	{
		a_c_p = *c;
		c_a_p = *a;
	}
		
	float* a1 = *a_c_p;
	float* c1 = *c_a_p;

	if (w[iGID] != 0)
	{
		//c[iGID] = iGID - b[2];
		int wtv1 = iGID - b[2];
		int wtv2 = iGID + b[2];
		int wtv3 = iGID - 1;
		int wtv4 = iGID + 1;
		c1[iGID] = b[1]
			* a1[iGID]
			+ b[0]
			* (
			a1[wtv1]
			+ a1[wtv2]
			+ a1[wtv3]
			+ a1[wtv4]
			);
	}
	else
		c[iGID] = -1;

	// add the vector elements
	//c[iGID] = iGID;// a[iGID] + b[iGID

	/*matFinale[i][j] = one_minus_four_times_tdhh
		* matInitiale[i][j]
		+ td_div_h_squ
		* (matInitiale[i - 1][j] + matInitiale[i + 1][j] + matInitiale[i][j - 1] + matInitiale[i][j + 1]);*/

}