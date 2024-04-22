#include <arrayfire.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <ostream>
#include <vector>
#include <af/internal.h>
#include <af/array.h>
#include <iomanip>
#include <string>
#include <chrono>
#include <omp.h>
using namespace std::chrono;

// This constant determines which version of the library your client code sees,
// and should be set (if needed) before including RNifti.h. The default is 1.
#define RNIFTI_NIFTILIB_VERSION 2
#include "RNifti.h"
#include "npy.hpp"

using namespace af;
using namespace std;


void kernel_to_matrix(vector<float> kernel, int dilation, af::array &row_indices, af::array &col_indices, 
                      af::array &values, int width, int height);

void kernel_to_matrix(vector<float> kernel, int dilation, af::array &row_indices, af::array &col_indices, 
                      af::array &values, int width, int height){

    int kernel_dim = 3;
    int total_channels = kernel.size()/27;

    vector<int> x_ind, y_ind;
    vector<float> val;

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {

            vector<float> kernel_value;
            
            for(int i = 0; i < total_channels; i++){
                for (int z = 0; z < kernel_dim; z++){
                    kernel_value.push_back(kernel[((z * 9) + ((y + 1) * 3) + (x + 1)) + i * 27]);
                }
            }

            // Loop through each position in the output slice (256x256)
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    // Calculate the corresponding position in the input with dilation
                    int input_i = i + y * dilation;
                    int input_j = j + x * dilation;

                    // Ensure the position does not fall into the virtual padding
                    if (0 <= input_i && input_i < height && 0 <= input_j && input_j < width) {
                        // Calculate the 1D index for the input
                        int input_idx = input_i * width + input_j;
                        // Calculate the 1D index for the matrix
                        int mat_idx = i * width + j;
                        // Append indices and kernel value to the lists
                        x_ind.push_back(mat_idx);
                        y_ind.push_back(input_idx);
                        val.insert(val.end(), kernel_value.begin(), kernel_value.end());
                    }
                }
            }
        }
    } 

    row_indices = af::array(dim4(x_ind.size()), x_ind.data());   
    col_indices = af::array(dim4(y_ind.size()), y_ind.data());   

    values = af::array(dim4(3, val.size()/3), val.data()); 
    values = af::transpose(values);
    values = af::flat(values);
    values = af::moddims(values, dim4(total_channels, val.size()/total_channels));
    values = af::transpose(values);
}


af::array generate_sparse_matrix(af::array values, af::array row_indices, af::array col_indices, 
                                 int out_slice_dims){

    af::array sparse_matrix_coo = af::sparse(out_slice_dims, out_slice_dims, values, row_indices, col_indices, AF_STORAGE_COO);
    
    af::array sparse_matrix_csr = sparseConvertTo(sparse_matrix_coo, AF_STORAGE_CSR);

    return sparse_matrix_csr;
}


af::array convolve3d(af::array &signal, af::array &row_indices, af::array &col_indices, 
                     af::array values, int out_channels, int dilation){

    int width = 256;
    int height = 256;
    int depth = 256;
    int out_slice_dims = width * height;
    
    af::array output = constant(0,  dim4(depth * width * height));
    af::array sparse_matrix_middle, sparse_matrix_first, sparse_matrix_last;

    for(int i = 0; i < out_channels; i++){
        int tmp = row_indices.dims()[0];

        sparse_matrix_first = generate_sparse_matrix(values(seq(tmp), i), row_indices, col_indices, out_slice_dims);
        sparse_matrix_middle = generate_sparse_matrix(values(seq(tmp, 2*tmp-1), i), row_indices, col_indices, out_slice_dims);
        sparse_matrix_last = generate_sparse_matrix(values(seq(2*tmp, values.dims()[0]-1), i), row_indices, col_indices, out_slice_dims);
        
        #pragma omp parallel for
        for(int j = 0; j < depth; j++){
            af::seq slice(j * out_slice_dims, (j + 1) * out_slice_dims - 1);
            output(slice) += matmul(sparse_matrix_middle, signal(slice, i));
            output.eval();

            if(j < depth - dilation){
                output(operator+(slice, dilation * out_slice_dims)) += matmul(sparse_matrix_first, signal(slice, i));    
                output.eval();
            }
            
            if(j >= dilation && j < depth - dilation){
                output(operator-(slice, dilation * out_slice_dims)) += matmul(sparse_matrix_last, signal(slice, i));  
                output.eval();  
            }
        }
    }
        
    return output;
}

af::array ELU(af::array signal){
    return af::expm1(signal * (signal <= 0.0)) + signal * (signal > 0.0);
}



af::array meshnet(af::array &signal, int n_layers){
    af::array filter, bias, row_indices, col_indices, values;
    af::array output(dim4(256 * 256 * 256, 5));
    const int dilation[] = {1, 2, 4, 8, 16, 8, 4, 2, 1};
    int in_channels, out_channels, first_index, last_index;
    string path;

    for(int i = 0; i < n_layers; i++){
       output = constant(0, dim4(256 * 256 * 256, 5));

       path = "../../5chan_wb/5chan_layer0" + to_string(i);

       in_channels = (i == 0) ? 1 : 5;
       out_channels = (i == n_layers-1) ? 3 : 5;

       auto f = npy::read_npy<float>(path + "w.npy");
       vector<float> filter = f.data;

       auto b = npy::read_npy<float>(path + "b.npy");
       bias = af::array(1, out_channels, 1, 1, (b.data).data());

       if(i == n_layers - 1){
        output = matmul(signal, af::array(5, 3, 1, filter.data()));
        output = ELU(output + bias);

        af::array max_index, max_value;
        af::max(max_value, max_index, af::reorder(af::moddims(output, dim4(256, 256, 256, out_channels)), 1, 2, 0, 3), 3);

        return max_index;
       }
        
       kernel_to_matrix(filter, dilation[i], row_indices, col_indices, values, 256, 256);
            
       for(int j = 0; j < out_channels; j++){
            output(span, j) = convolve3d(signal, row_indices, col_indices, 
                                         values(span, seq(in_channels * j, in_channels * (1 + j) - 1)), 
                                         in_channels, dilation[i]);
       }
       
       output = ELU(output + bias);
       signal = output;
    }
    return output;
}

void preprocess(af::array &signal, float lower_quantile = 0.05, float upper_quantile = 0.95){

    const af::array sorted_signal = af::sort(signal);

    int n_elements = sorted_signal.elements();

    int lower_index = std::floor(n_elements * lower_quantile);
    int upper_index = std::ceil(n_elements * upper_quantile) - 1;

    float val_lower_index = sorted_signal(lower_index).scalar<float>();
    float val_upper_index = sorted_signal(upper_index).scalar<float>();

    signal = (signal - val_lower_index)/(val_upper_index - val_lower_index);
}

int main(){

    auto start = high_resolution_clock::now();

    RNifti::NiftiImage image("../../quantile05_95_normalized_t1_c.nii.gz");

    RNifti::NiftiImageData niidata = image.data();

    vector<float> sig(niidata.begin(), niidata.end());
    af::array signal = af::array(dim4(256, 256, 256), sig.data());
    signal = af::reorder(signal, 2, 1, 0, 3);
    signal = af::flat(signal);

    // preprocess(signal, 0.05, 0.95);

    af::array output = meshnet(signal, 10);
    output = af::reorder(output, 1, 0, 2, 3);


    vector<int> out(256 * 256 * 256);
    output.host(out.data());
    image.replaceData(out, DT_UINT8);
    image.toFile("../../ab.nii.gz", "auto", -1);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total Execution Time:\t"<<duration.count() << "\n";
}