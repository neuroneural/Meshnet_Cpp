// 1. Add includes
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

static float scalar(float val) {
    return float(val);
}

af::array convolve3d(af::array const &signal, af::array const &filter,
                   const dim4 strides, const dim4 padding, const dim4 dilation);

af::array convolve3d_unwrap(const af::array &signal, const af::array &filter,
                          const dim4 &strides, const dim4 &padding,
                          const dim4 &dilation, const dim4 &filter_dims);

af::array unwrap3d(const af::array &in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const dim_t nx, const dim_t ny, const bool is_column);

void unwrap3d_dim(af::array &out, const af::array &in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const int d);

af::array meshnet(af::array &signal, int layers);

void unwrap3d_dim(af::array &out, const af::array &in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const dim_t nx, const dim_t ny, const int d) {

    auto start = high_resolution_clock::now();
    const float *inPtr = in.device<float>();
    float *outPtr      = out.device<float>();

    dim4 idims    = in.dims();
    dim4 odims    = out.dims();
  
    dim4 istrides = af::getStrides(in);
    dim4 ostrides = af::getStrides(out);

    for (dim_t w = 0; w < odims[3]; w++) {
        for (dim_t v = 0; v < odims[2]; v++) {
            dim_t cOut    = w * ostrides[3] + v * ostrides[2];
            dim_t cIn     = w * istrides[3] + v * istrides[2];
            
            const float *iptr = inPtr + cIn;
            float *optr_      = outPtr + cOut;
            
            #pragma omp parallel for
            for (dim_t col = 0; col < odims[d]; col++) {
                // Offset output ptr
                // cout<<("Hello World... from thread =")<<omp_get_num_threads()<<"\n";
                float *optr = optr_ + col * ostrides[d];
                // Calculate input window index
                dim_t winz = col / (nx * ny);      
                dim_t winy = (col % (nx * ny)) / nx; 
                dim_t winx = (col % (nx * ny)) % nx;
                
                dim_t startx = winx * sx;
                dim_t starty = winy * sy;
                dim_t startz = winz * sz;

                dim_t spx = startx - px;
                dim_t spy = starty - py;
                dim_t spz = startz - pz;

                // Short cut condition ensuring all values within input
                // dimensions
                bool cond = (spx >= 0 && spx + (wx * dx) < idims[0] &&
                             spy >= 0 && spy + (wy * dy) < idims[1] &&
                             spz >= 0 && spz + (wz * dz) < idims[2]);

                for(dim_t z = 0; z < wz; z++) {
                    dim_t zpad = spz + z * dz;
                    for (dim_t y = 0; y < wy; y++) {
                        dim_t ypad = spy + y * dy;
                        for (dim_t x = 0; x < wx; x++) {
                            dim_t xpad = spx + x * dx;

                            dim_t oloc = (z * wy * wx + y * wx + x);
                            if (d == 0) oloc *= ostrides[1];

                            if (cond || (xpad >= 0 && xpad < idims[0] &&
                                         ypad >= 0 && ypad < idims[1] &&
                                         zpad >= 0 && zpad < idims[2])) {
                                dim_t iloc =
                                    (zpad * istrides[2] + ypad * istrides[1] + xpad * istrides[0]);
                                optr[oloc] = iptr[iloc];
                            } else {
                                optr[oloc] = scalar(0.0f);
                            }
                            
                        }
                    }
                }
            }
        }
    }
    in.unlock();
    out.unlock();
     auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Unwrap Execution Time:\t"<<duration.count() << "\n";
}

af::array unwrap3d(const af::array &in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const dim_t nx,
                const dim_t ny, const dim_t nz, const bool is_column) {

    af::dim4 odims(wx * wy * wz, nx * ny * nz, 1, (in.dims())[3]);

    if (!is_column) { std::swap(odims[0], odims[1]); }
    
    af::array outArray = af::array(odims);
    const int d = (is_column) ? 1 : 0;
    auto start = high_resolution_clock::now();
    unwrap3d_dim(outArray, in, wx, wy, wz, sx, sy, sz, px,
                       py, pz, dx, dy, dz, nx, ny, d);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Unwrap Execution Time:\t"<<duration.count() << "\n";
    return outArray;
}

af::array convolve3d_unwrap(const af::array &signal, const af::array &filter,
                          const dim4 &strides, const dim4 &padding,
                          const dim4 &dilation, const dim4 &filter_dims) {
    dim4 sDims = signal.dims();
    dim4 fDims = filter_dims;

    dim_t outputWidth =
        1 + (sDims[0] + 2 * padding[0] - (((fDims[0] - 1) * dilation[0]) + 1)) /
                strides[0];
    dim_t outputHeight =
        1 + (sDims[1] + 2 * padding[1] - (((fDims[1] - 1) * dilation[1]) + 1)) /
                strides[1];
    dim_t outputDepth =
        1 + (sDims[2] + 2 * padding[2] - (((fDims[2] - 1) * dilation[2]) + 1)) /
                strides[2];
    
    const bool retCols = false;
    
    af::array unwrapped =
        unwrap3d(signal, fDims[0], fDims[1], fDims[2], strides[0], strides[1], strides[2], padding[0],
               padding[1], padding[2], dilation[0], dilation[1], dilation[2], outputWidth, outputHeight, outputDepth, retCols);

    unwrapped  = af::reorder(unwrapped, 1, 3, 0, 2);
    dim4 uDims = unwrapped.dims();
 
    unwrapped =
        af::moddims(unwrapped, dim4(uDims[0] * uDims[1], uDims[2] * uDims[3]));

    af::array res =
        matmul(unwrapped, filter, AF_MAT_TRANS, AF_MAT_NONE);   
            
    res = af::moddims(res, dim4(outputWidth, outputHeight, outputDepth, filter.dims()[1]));
    return res;
}

af::array ELU(af::array &signal){
    return af::operator+(af::exp(af::operator*(signal, af::operator<=(signal, 0.0)))-1, af::operator*(signal, af::operator>(signal, 0.0)));
}


af::array convolve3d(af::array const &signal, af::array const &filter,
                   const dim4 strides, const dim4 padding, const dim4 dilation, const dim4 filter_dims) {
    af::array out = af::array(dim4());
    out = convolve3d_unwrap(signal, filter, strides, padding, dilation, filter_dims);
    
    return out;
}

af::array lastlayer_argmax(const af::array &signal, const af::array &filter, const af::array &bias,
                          const dim4 &strides, const dim4 &padding,
                          const dim4 &dilation, const dim4 &filter_dims) {
    dim4 fDims = filter.dims();
    af::array track_max, track_indx, convolved;
    track_indx = constant(0, 256, 256, 256, u8);
    int nfilt = fDims[1];
    int nchl = fDims[2];

    for(int i = 0; i<nfilt; i++){
        convolved = convolve3d(signal(span, span, span, 0), filter(span, i, 0), strides, padding, dilation, filter_dims);       
        convolved.eval();
        for(int j = 1; j<nchl; j++){
            convolved = af::operator+(convolved, convolve3d(signal(span, span, span, j), filter(span, i, j), strides, padding, dilation, filter_dims));
            convolved.eval();
        }
        convolved = af::operator+(convolved, bias(span, span, span, i));
        convolved.eval();
        convolved = ELU(convolved);
        convolved.eval();
        if(i==0){
            track_max = convolved;
            cout<<"Channel_0"<<"\n";
        }
        else{
            cout<<"Channel_"<<i<<"\n";
            track_indx = af::max(track_indx, i*af::operator<(track_max, convolved));
            track_indx.eval();
            track_max = af::max(track_max, convolved);
            track_max.eval();
        }       
    }

    return track_indx;
}

af::array meshnet(af::array &signal, int layers){
    af::array filter, bias, convolved;
    int dp[] = {1, 2, 4, 8, 16, 8, 4, 2, 1};
    dim4 filter_dims(3, 3, 3), strides(1, 1, 1), dilation, padding;
    int undim, val, chdim, nfdim;
    string path;
    
    for(int i = 0; i<layers; i++)
    {
        if(i>0 && i<9) continue;
       val = 1, chdim = 5, undim = 27, nfdim = 5, path = "";
        dilation = padding = {dp[i], dp[i], dp[i]}, path = "../../model_weights_unwrapped_5chan/5chan_layer0" + to_string(i);
        if(i==0){
            chdim = 1;
        }
       if(i==layers-1){
        filter_dims = {1, 1, 1}, undim = 1, nfdim = 3, dilation = {1, 1, 1}, padding = {0, 0, 0};
       }

       auto f = npy::read_npy<float>(path+"w.npy");
       filter = af::array(undim, nfdim, chdim, (f.data).data());
       
       auto b = npy::read_npy<float>(path+"b.npy");
       bias = af::array(1, 1, 1, nfdim, (b.data).data());

       cout<<"Layer_"<<i<<"\nSignal Dimensions\t"<<(signal.dims())<<"\n"<<"Filter Dimensions\t"<<(filter.dims())<<"\n";

       if(i == 0){
        convolved = convolve3d(signal, filter, strides, padding, dilation, filter_dims);
        convolved.eval();
        convolved = af::operator+(convolved, bias);
        convolved.eval();
        convolved = ELU(convolved);
        convolved.eval();
       } 
       else if(0<i && i<9){
        convolved = convolve3d(signal(span, span, span, 0), filter(span, span, 0), strides, padding, dilation, filter_dims);
        cout<<"Channel_0"<<"\n";
        convolved.eval();
        for(int j = 1; j<chdim; j++){
            convolved = af::operator+(convolved, convolve3d(signal(span, span, span, j), filter(span, span, j), strides, padding, dilation, filter_dims));
            cout<<"Channel_"<<j<<"\n";
            convolved.eval();
        }
        convolved = af::operator+(convolved, bias);
        convolved.eval();
        convolved = ELU(convolved);
        convolved.eval();
       }
       else{
        convolved = lastlayer_argmax(convolved, filter, bias, strides, padding, dilation, filter_dims);
       }   
       cout<<"Output Dimension\t"<<(convolved.dims())<<"\n";
       signal = convolved;
    }   

    return convolved;
}

af::array preprocess(af::array &signal, const float lowerQuantile = 0.01, const float upperQuantile = 0.99){
    const af::array fsort = af::sort(af::flat(signal));
    const dim_t numElements = fsort.dims()[0];
    const dim_t lidx = std::floor(numElements * lowerQuantile);
    const dim_t uidx = std::ceil(numElements * upperQuantile) - 1;
    const af::array qmin = fsort(lidx);
    signal = (signal-qmin)/(fsort(uidx)-qmin);
    signal.eval();
    return signal;
}

int main(int argc, char** argv){

    af::info();
    auto start = high_resolution_clock::now();
    RNifti::NiftiImage image("../../t1_c.nii.gz");
    RNifti::NiftiImageData niidata = image.data();
    vector<float> sig(niidata.begin(), niidata.end());
    af::array signal = af::array(256, 256, 256, 1, sig.data());
    signal = af::reorder(signal, 1, 0, 2, 3);

    signal = preprocess(signal, 0.01, 0.99);
    
    af::array output = meshnet(signal, 10);
    output = af::reorder(output, 1, 0, 2, 3);
    af_print(output);

    vector<int> out(256*256*256);
    output.host(out.data());
    image.replaceData(out, DT_UINT8);
    image.toFile("../../5chan_out.nii.gz", "auto", -1);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total Execution Time:\t"<<duration.count() << "\n";

    return 0;
}
