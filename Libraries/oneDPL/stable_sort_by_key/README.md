# `Stable sort by key` Sample

Stable sort by key is a sorting operation when sorting two sequences (keys and values). Only keys are compared, but both keys and values are swapped. This sample demonstrates `counting_iterator` and `zip_iterator` from Intel&reg; oneAPI DPC++ Library (oneDPL).


| Optimized for                   | Description                                                                      |
|---------------------------------|----------------------------------------------------------------------------------|
| OS                              | Linux* Ubuntu* 18.04                                                             |
| Hardware                        | Skylake with GEN9 or newer                                                       |
| Software                        | Intel&reg; oneAPI DPC++/C++ Compiler; Intel&reg; oneAPI DPC++ Library (oneDPL)   |
| What you will learn             | How to use `counting_iterator` and `zip_iterator`                                |
| Time to complete                | At most 5 minutes                                                                |

## Purpose

The sample models stable sorting by key: during the sorting of 2 sequences (keys and values), only keys are compared, but keys and values are swapped.
It fills two buffers (one of the buffers is filled using `counting_iterator`) and then sorts them using `zip_iterator`.

The sample demonstrates how to use `counting_iterator` and `zip_iterator` using Intel&reg; oneAPI DPC++ Library (oneDPL).
* `counting_iterator` helps fill the sequence with the numbers zero through `n` using std::copy.
* `zip_iterator` provides the ability to iterate over several sequences simultaneously.

## Key Implementation Details

Following Parallel STL algorithms are used in the code: `transform`, `copy`, `stable_sort`.

`counting_iterator`, `zip_iterator` are used from the Extension API of oneDPL.

## License

Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program Licenses can be found here: [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt)

## Building the 'Stable sort by key' Program for CPU and GPU

### Running Samples In DevCloud
If running a sample in the Intel DevCloud, remember that you must specify the compute node (CPU, GPU, FPGA) and whether to run in batch or interactive mode. For more information, see the Intel&reg; oneAPI Base Toolkit Get Started Guide (https://devcloud.intel.com/oneapi/get-started/base-toolkit/)

### Using Visual Studio Code*  (Optional)

You can use Visual Studio Code (VS Code) extensions to set your environment, create launch configurations,
and browse and download samples.

The basic steps to build and run a sample using VS Code include:
 - Download a sample using the extension **Code Sample Browser for Intel oneAPI Toolkits**.
 - Configure the oneAPI environment with the extension **Environment Configurator for Intel oneAPI Toolkits**.
 - Open a Terminal in VS Code (**Terminal>New Terminal**).
 - Run the sample in the VS Code terminal using the instructions below.

To learn more about the extensions and how to configure the oneAPI environment, see
[Using Visual Studio Code with Intel® oneAPI Toolkits](https://software.intel.com/content/www/us/en/develop/documentation/using-vs-code-with-intel-oneapi/top.html).

After learning how to use the extensions for Intel oneAPI Toolkits, return to this readme for instructions on how to build and run a sample.

### On a Linux* System
Perform the following steps:

1. Build the program using the following `cmake` commands.
```
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
```

2. Run the program:
```
    $ make run
```

3. Clean the program using:
```
    $ make clean
```

## Running the Sample
### Example of Output

```
success
Run on Intel(R) Gen9
```
