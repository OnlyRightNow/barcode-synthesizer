# Barcode Synthesizer

<a href="https://github.com/OnlyRightNow/barcode-synthesizer/blob/master/LICENSE.md"><img src="https://img.shields.io/pypi/l/python-barcode.svg" title="License" alt="License"></a>
  
Creates a set of realistically looking barcode images and a textfile containing the values.

## Motivation
There is a lack of useful barcodes datasets e.g. for training a barcode-decoder network.<br /> 
So I started writing a script that creates such a dataset with a barcode generator and 
rotates the barcodes in the 3D-plane, adds noise, changes brightness and intensity 
and adds blurring.<br /> 
The goal is to reduce the reality gap as much as possible.

<a href="https://github.com/OnlyRightNow/barcode-synthesizer/blob/master/example.png"><img src="https://github.com/OnlyRightNow/barcode-synthesizer/blob/master/output/example.png?raw=true" title="example" alt="example"></a>


## Requirements
- Python 3.5 or above
- [python-barcode](https://github.com/WhyNotHugo/python-barcode)

## Supported Barcodes
* EAN-8
* EAN-13
* EAN-14
* UPC-A
* JAN
* ISBN-10
* ISBN-13
* ISSN
* Code 39
* Code 128
* PZN

## Ideas?
If you think there is something missing or something is not working, [let me know](https://github.com/OnlyRightNow/barcode-synthesizer/issues).