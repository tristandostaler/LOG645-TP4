/*__kernel void HeatTransfer()
{
	int id = get_global_id(0);
}*/

__kernel void HeatTransfer(__global float* a, __global const int* w, __global const float* b, __global float* c)
{
	// get index into global data array
	int iGID = get_global_id(0);

	if (b[4] == 0)
	{
		if (w[iGID] != 0)
		{
			//c[iGID] = iGID - b[2];
			int wtv1 = iGID - b[2];
			int wtv2 = iGID + b[2];
			int wtv3 = iGID - 1;
			int wtv4 = iGID + 1;
			c[iGID] = b[1]
				* a[iGID]
				+ b[0]
				* (
				a[wtv1]
				+ a[wtv2]
				+ a[wtv3]
				+ a[wtv4]
				);
		}
		else
			c[iGID] = 0;
	}
	else
	{
		if (w[iGID] != 0)
		{
			//c[iGID] = iGID - b[2];
			int wtv1 = iGID - b[2];
			int wtv2 = iGID + b[2];
			int wtv3 = iGID - 1;
			int wtv4 = iGID + 1;
			a[iGID] = b[1]
				* c[iGID]
				+ b[0]
				* (
				c[wtv1]
				+ c[wtv2]
				+ c[wtv3]
				+ c[wtv4]
				);
		}
		else
			a[iGID] = 0;
	}
	

	// add the vector elements
	//c[iGID] = iGID;// a[iGID] + b[iGID

	/*matFinale[i][j] = one_minus_four_times_tdhh
		* matInitiale[i][j]
		+ td_div_h_squ
		* (matInitiale[i - 1][j] + matInitiale[i + 1][j] + matInitiale[i][j - 1] + matInitiale[i][j + 1]);*/

}