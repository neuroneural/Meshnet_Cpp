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

af::array convolve3d(const af::array &signal, const af::array &filter,
                          const dim4 &strides, const dim4 &padding,
                          const dim4 &dilation, const dim4 &filter_dims);

void unwrap3d_dim(af::array &out, const af::array &in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const dim_t nx, const dim_t ny);

af::array meshnet(af::array &signal, int layers);

static float scalar(float val) {
    return float(val);
}

void unwrap3d_dim(af::array &out, const af::array &in, const dim_t wx, const dim_t wy, const dim_t wz,
                const dim_t sx, const dim_t sy, const dim_t sz, const dim_t px, const dim_t py,
                const dim_t pz, const dim_t dx, const dim_t dy, const dim_t dz, const dim_t nx, const dim_t ny, float *inPtr, float *outPtr) {

    // inPtr = in.device<float>();
    // outPtr = out.device<float>();
    // cout<<inPtr<<outPtr;
    cout<<*inPtr;
    if(out.dims()[1]==1)
    {inPtr = in.device<float>();
    outPtr = out.device<float>();}

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
            // af_print(in(0,0,0,0));
            
            #pragma omp parallel for
            for (dim_t col = 0; col < odims[0]; col++) {

                // Offset output ptr
                float *optr = optr_ + col * ostrides[0];

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
                            oloc *= ostrides[1];

                            if (cond || (xpad >= 0 && xpad < idims[0] &&
                                         ypad >= 0 && ypad < idims[1] &&
                                         zpad >= 0 && zpad < idims[2])) {
                                dim_t iloc =
                                    (zpad * istrides[2] + ypad * istrides[1] + xpad * istrides[0]);
                                optr[oloc] = iptr[iloc];
                            } 
                            else {
                                optr[oloc] = scalar(0.0f);
                            }
                        }
                    }
                }
            }
        }
    }
    // in.unlock();
    // out.unlock();
}

af::array convolve3d(const af::array &signal, const af::array &filter,
                          const dim4 &strides, const dim4 &padding,
                          const dim4 &dilation, const dim4 &filter_dims, 
                          af::array &outArray, float *inPtr, float *outPtr) {
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

    auto start = high_resolution_clock::now();
    unwrap3d_dim(outArray, signal, fDims[0], fDims[1], fDims[2], strides[0], strides[1], strides[2], padding[0],
               padding[1], padding[2], dilation[0], dilation[1], dilation[2], outputWidth, outputHeight, inPtr, outPtr);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Unwrap Execution Time:\t"<<duration.count() << "\n";

    af::array unwrapped  = af::reorder(outArray, 1, 3, 0, 2);
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

af::array lastlayer_argmax(const af::array &signal, const af::array &filter, const af::array &bias,
                          const dim4 &strides, const dim4 &padding,
                          const dim4 &dilation, const dim4 &filter_dims, af::array &outArray, float *inPtr, float *outPtr) {
    dim4 fDims = filter.dims();
    af::array track_max, convolved;
    af::array track_indx = constant(0, 256, 256, 256, u8);
    int nfilt = fDims[1];
    int nchl = fDims[2];

    for(int i = 0; i<nfilt; i++){
        convolved = convolve3d(signal(span, span, span, 0), filter(span, i, 0), strides, padding, dilation, filter_dims, outArray, inPtr, outPtr);     
        for(int j = 1; j<nchl; j++){
            convolved = af::operator+(convolved, convolve3d(signal(span, span, span, j), filter(span, i, j), strides, padding, dilation, filter_dims, outArray, inPtr, outPtr));
            af::eval(convolved);
        }        
        convolved = af::operator+(convolved, bias(span, span, span, i));
        af::eval(convolved);
        convolved = ELU(convolved);
        if(i==0){
            track_max = convolved;
            // cout<<"Channel_0"<<"\n";
        }
        else{
            // cout<<"Channel_"<<i<<"\n";
            track_indx = af::max(track_indx, i*af::operator<(track_max, convolved));
            track_indx.eval();
            track_max = af::max(track_max, convolved);
            track_max.eval();
        }       
    }
    return track_indx;
}


af::array meshnet(af::array &signal, int layers){
    af::array filter, bias, filter0,bias0,filter9,bias9, convolved;
    float *outPtr, *inPtr;
    af::dim4 sdims = signal.dims();
    dim4 filter_dims(3, 3, 3), strides(1, 1, 1), dilation, padding;
    af::dim4 odims(sdims[0] * sdims[1] * sdims[2], filter_dims[0] * filter_dims[1] * filter_dims[2], 1, 1);
    af::array outArray = af::array(odims);
    int dp[] = {1, 2, 4, 8, 16, 8, 4, 2, 1};
    int undim, val, chdim, nfdim;
    string path;
    af::array in(256,256,256,1);

    inPtr = signal.device<float>();
    outPtr = outArray.device<float>();
    cout<<inPtr<<outPtr;

    
    for(int i = 0; i<layers; i++)
    {   
        val = 1, chdim = 5, undim = 27, nfdim = 5, path = "";
        dilation = padding = {dp[i], dp[i], dp[i]}, path = "../../model_weights_unwrapped_5chan/5chan_layer0" + to_string(i);
       if(i==0){
            chdim = 1;
            auto f = npy::read_npy<float>("../../model_weights_unwrapped_5chan/5chan_layer00w.npy");
            filter0 = af::array(undim, nfdim, chdim, (f.data).data());
            auto b = npy::read_npy<float>("../../model_weights_unwrapped_5chan/5chan_layer00b.npy");
            bias0 = af::array(1, 1, 1, nfdim, (b.data).data());
        }
       if(i==layers-1){
        filter_dims = {1, 1, 1}, undim = 1, nfdim = 3, dilation = {1, 1, 1}, padding = {0, 0, 0};
        auto f = npy::read_npy<float>("../../model_weights_unwrapped_5chan/5chan_layer09w.npy");
        filter9 = af::array(undim, nfdim, chdim, (f.data).data());
        auto b = npy::read_npy<float>("../../model_weights_unwrapped_5chan/5chan_layer09b.npy");
        bias9 = af::array(1, 1, 1, nfdim, (b.data).data());
       }  
       
       
       if(i==1){
       auto f = npy::read_npy<float>("../../model_weights_unwrapped_5chan/weights.npy");
      filter = af::array(undim, nfdim, chdim, 8, (f.data).data());

       auto b = npy::read_npy<float>("../../model_weights_unwrapped_5chan/bias.npy");
       bias = af::array(nfdim, 1, 1, 8, (b.data).data());
       }

       if(i == 0){
        convolved = convolve3d(signal(span, span, span, 0), filter0, strides, padding, dilation, filter_dims, outArray, inPtr, outPtr);
        convolved = af::operator+(convolved, bias0);
        convolved = ELU(convolved);
       } 
       else if(0<i && i<9){
        for(int j = 0; j<chdim; j++){
            if(j == 0)
            { 
                convolved(span,span,span,span) = convolve3d(signal(span, span, span, 0), filter(span, span, j, i-1), strides, padding, dilation, filter_dims, outArray, inPtr+256*256*256*j, outPtr); 
                af::sync();
                continue; 
            }
            af::array s = convolved(span,span,span,span);
            convolved(span,span,span,span) = af::operator+(s, convolve3d(signal(span, span, span, j), filter(span, span, j, i-1), strides, padding, dilation, filter_dims, outArray, inPtr+256*256*256*j, outPtr));
            af::sync();
        }
        convolved.eval();
        convolved(span,span,span,span) = af::operator+(convolved(span,span,span,span), af::moddims(bias(span,span,span,i-1),1,1,1,5));
        
        convolved(span,span,span,span) = ELU(convolved);
       }
       else{
        af::array outArray1 = af::array(sdims[0] * sdims[1] * sdims[2], filter_dims[0] * filter_dims[1] * filter_dims[2], 1, 1);
        convolved = lastlayer_argmax(signal, filter9, bias9, strides, padding, dilation, filter_dims, outArray1, inPtr, outPtr);
       }     
       if(i!=layers-1)
       signal(span,span,span,span) = convolved(span,span,span,span);
       auto start = high_resolution_clock::now();
       af::sync();
       auto stop = high_resolution_clock::now();
       auto duration = duration_cast<microseconds>(stop - start);
       cout << "sync Execution Time:\t"<<duration.count() << "\n";
       cout<<"Layer_"<<i<<"\n";
    }   

    return convolved;
}

af::array preprocess(af::array signal, const float lowerQuantile = 0.01, const float upperQuantile = 0.99){
    const af::array fsort = af::sort(af::flat(signal));
    const dim_t numElements = fsort.dims()[0];
    const dim_t lidx = std::floor(numElements * lowerQuantile);
    const dim_t uidx = std::ceil(numElements * upperQuantile) - 1;
    const af::array qmin = fsort(lidx);
    signal = (signal-qmin)/(fsort(uidx)-qmin);
    return signal;
}

int main(int argc, char** argv){

    // af::info();
    // af::array test = randu(5,5,5,3);
    // float *ptr = test.device<float>();
    // cout<<ptr;
    // af::array test1 = randu(5,5,5,3);   
    // float *ptr1 = test1.device<float>();
    // cout<<ptr1;
    // test= test1;
    // af::sync();
    // ptr = test.device<float>();
    // cout<<ptr;
    // ptr1 = test1.device<float>();
    // cout<<ptr1;
    // af_print(test);

    // cout<<*(ptr+0);
    // cout<<*(ptr+125);
    // cout<<*(ptr+125+125);
    // cout<<*(ptr+0);
    // cout<<*(ptr+125);
    // cout<<*(ptr+);
   

    auto start = high_resolution_clock::now();
    RNifti::NiftiImage image("../../t1_c.nii.gz");
    RNifti::NiftiImageData niidata = image.data();
    vector<float> sig(niidata.begin(), niidata.end());
    af::array signal = af::array(256, 256, 256, 5);
    af::array signal1 = af::array(256, 256, 256, 1, sig.data());
    signal1 = af::reorder(signal1, 1, 0, 2, 3);
    signal(span,span,span,0) = preprocess(signal1, 0.01, 0.99);
    // af_print(signal(span,span,span,1));
  
    af::array output = af::array(256, 256, 256, u8);
    output = meshnet(signal, 10);
    output.eval();
    output = af::reorder(output, 1, 0, 2, 3);
    // af_print(output);

    vector<int> out(256*256*256);
    output.host(out.data());
    image.replaceData(out, DT_UINT8);
    image.toFile("../../5chan_out1.nii.gz", "auto", -1);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Total Execution Time:\t"<<duration.count() << "\n";

    return 0;
}
