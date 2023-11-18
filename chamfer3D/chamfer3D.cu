
#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__ void NmDistanceKernel2(int b,int n,const float * xyz, int m,const float * xyz2,const float* L, float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*3];
	__shared__ float bufL[batch*6];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
                    buf[j] = xyz2[(i * m + k2) * 3 + j];
			}
			for (int j=threadIdx.x;j<end_k*6;j+=blockDim.x){
                    bufL[j] = L[(i * m + k2) * 3 + j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*3+0];
				float y1=xyz[(i*n+j)*3+1];
				float z1=xyz[(i*n+j)*3+2];
				int best_i=0;
				float best=0;
				#pragma unroll
				for (int k=0;k<end_k;k++){
						float x2=buf[k*3+0]-x1;
						float y2=buf[k*3+1]-y1;
						float z2=buf[k*3+2]-z1;
						float a = bufL[k*6+0]; // L[i, j, 0, 0]
						float b = bufL[k*6+1]; // L[i, j, 1, 0]
						float c = bufL[k*6+2]; // L[i, j, 1, 1]
						float d = bufL[k*6+3]; // L[i, j, 2, 0]
						float e = bufL[k*6+4]; // L[i, j, 2, 1]
						float f = bufL[k*6+5]; // L[i, j, 2, 2]
						float v1 = x2 / a;
						float v2 = (a * y2 - b * x2) / (a * c);
						float v3 = (a * c * z2 - a * e * y2 + b * e * x2 - c * d * x2) / (a * c * f);
						float dist = v1*v1 + v2*v2 + v3*v3;
						if (k==0 || dist<best){
							best=dist;
							best_i=k+k2;
						}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}

__global__ void NmDistanceKernel1(int b,int n,const float * xyz,const float* L, int m,const float * xyz2,float * result,int * result_i){
	const int batch=512;
	__shared__ float buf[batch*3];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*3;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*3+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*3+0];
				float y1=xyz[(i*n+j)*3+1];
				float z1=xyz[(i*n+j)*3+2];
				float a = L[(i * n + j) * 6 + 0]; // L[i, j, 0, 0]
                float b = L[(i * n + j) * 6 + 1]; // L[i, j, 1, 0]
                float c = L[(i * n + j) * 6 + 2]; // L[i, j, 1, 1]
                float d = L[(i * n + j) * 6 + 3]; // L[i, j, 2, 0]
                float e = L[(i * n + j) * 6 + 4]; // L[i, j, 2, 1]
                float f = L[(i * n + j) * 6 + 5]; // L[i, j, 2, 2]
				int best_i=0;
				float best=0;
				#pragma unroll
				for (int k=0;k<end_k;k++){
						float x2=buf[k*3+0]-x1;
						float y2=buf[k*3+1]-y1;
						float z2=buf[k*3+2]-z1;
						float v1 = x2 / a;
						float v2 = (a * y2 - b * x2) / (a * c);
						float v3 = (a * c * z2 - a * e * y2 + b * e * x2 - c * d * x2) / (a * c * f);
						float dist = v1*v1 + v2*v2 + v3*v3;
						if (k==0 || dist<best){
							best=dist;
							best_i=k+k2;
						}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}
// int chamfer_cuda_forward(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream){
int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2,at::Tensor L, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2){

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
	const auto m = xyz2.size(1); //num_points point cloud B

	NmDistanceKernel1<<<dim3(128,64,1),1024>>>(batch_size, n, xyz1.data<float>(), L.data<float>(), m, xyz2.data<float>(), dist1.data<float>(), idx1.data<int>());
	NmDistanceKernel2<<<dim3(128,64,1),1024>>>(batch_size, m, xyz2.data<float>(), n, xyz1.data<float>(),L.data<float>(), dist2.data<float>(), idx2.data<int>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;


}
__global__ void NmDistanceGradKernel1(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2, const float * L){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			float a = L[(i * n + j) * 6 + 0]; // L[i, j, 0, 0]
			float b = L[(i * n + j) * 6 + 1]; // L[i, j, 1, 0]
			float c = L[(i * n + j) * 6 + 2]; // L[i, j, 1, 1]
			float d = L[(i * n + j) * 6 + 3]; // L[i, j, 2, 0]
			float e = L[(i * n + j) * 6 + 4]; // L[i, j, 2, 1]
			float f = L[(i * n + j) * 6 + 5]; // L[i, j, 2, 2]
			int j2=idx1[i*n+j];
			float x2=x1 - xyz2[(i*m+j2)*3+0];
			float y2=y1 - xyz2[(i*m+j2)*3+1];
			float z2=y2 - xyz2[(i*m+j2)*3+2];
			float x3 = x2 / a;
			float y3 = (a * y2 - b * x2) / (a * c);
			float z3 = (a * c * z2 - a * e * y2 + b * e * x2 - c * d * x2) / (a * c * f);
			float x4 = z3 / f;
			float y4 = (y3 - e * x4) / c;
			float z4 = (x3 - b * y4 - d * x4) / a;
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*x4);
			atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*y4);
			atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*z4);
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),-(g*x4));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),-(g*y4));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),-(g*z4));
		}
	}
}

__global__ void NmDistanceGradKernel2(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2, const float * L){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*3+0];
			float y1=xyz1[(i*n+j)*3+1];
			float z1=xyz1[(i*n+j)*3+2];
			int j2=idx1[i*n+j];
			float a = L[(i * m + j2) * 6 + 0]; // L[i, j, 0, 0]
			float b = L[(i * m + j2) * 6 + 1]; // L[i, j, 1, 0]
			float c = L[(i * m + j2) * 6 + 2]; // L[i, j, 1, 1]
			float d = L[(i * m + j2) * 6 + 3]; // L[i, j, 2, 0]
			float e = L[(i * m + j2) * 6 + 4]; // L[i, j, 2, 1]
			float f = L[(i * m + j2) * 6 + 5]; // L[i, j, 2, 2]
			float x2=x1 - xyz2[(i*m+j2)*3+0];
			float y2=y1 - xyz2[(i*m+j2)*3+1];
			float z2=z1 - xyz2[(i*m+j2)*3+2];
			float x3 = x2 / a;
			float y3 = (a * y2 - b * x2) / (a * c);
			float z3 = (a * c * z2 - a * e * y2 + b * e * x2 - c * d * x2) / (a * c * f);
			float x4 = z3 / f;
			float y4 = (y3 - e * x4) / c;
			float z4 = (x3 - b * y4 - d * x4) / a;
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*3+0]),g*x4);
			atomicAdd(&(grad_xyz1[(i*n+j)*3+1]),g*y4);
			atomicAdd(&(grad_xyz1[(i*n+j)*3+2]),g*z4);
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+0]),-(g*x4));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+1]),-(g*y4));
			atomicAdd(&(grad_xyz2[(i*m+j2)*3+2]),-(g*z4));
		}
	}
}

// int chamfer_cuda_backward(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream){
int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2, at::Tensor L){
	// cudaMemset(grad_xyz1,0,b*n*3*4);
	// cudaMemset(grad_xyz2,0,b*m*3*4);
	
	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
	const auto m = xyz2.size(1); //num_points point cloud B

	NmDistanceGradKernel1<<<dim3(1,16,1),256>>>(batch_size,n,xyz1.data<float>(),m,xyz2.data<float>(),graddist1.data<float>(),idx1.data<int>(),gradxyz1.data<float>(),gradxyz2.data<float>(), L.data<float>());
	NmDistanceGradKernel2<<<dim3(1,16,1),256>>>(batch_size,m,xyz2.data<float>(),n,xyz1.data<float>(),graddist2.data<float>(),idx2.data<int>(),gradxyz2.data<float>(),gradxyz1.data<float>(), L.data<float>());
	
	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;
	
}

