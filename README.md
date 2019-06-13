## Barcode Synthesizer
Creates a set of realistically looking barcodes images and a textfile containing the values.

## Motivation
There is a lack of useful barcodes datasets e.g. for training a barcode-decoder network.<br /> 
So I started writing a script that creates such a dataset with a barcode generator and 
rotates the barcodes in the 3D-plane, adds noise, changes brightness and intensity 
and adds blurring.<br /> 
The goal is to reduce the reality gap as much as possible.
## Build status
Build status of continus integration i.e. travis, appveyor etc. Ex. - 

[![Build Status](https://travis-ci.org/akashnimare/foco.svg?branch=master)](https://travis-ci.org/akashnimare/foco)
[![Windows Build Status](https://ci.appveyor.com/api/projects/status/github/akashnimare/foco?branch=master&svg=true)](https://ci.appveyor.com/project/akashnimare/foco/branch/master)
 
## Screenshots
Include logo/demo screenshot etc.

## Requirements
- Python 3.5 or above
- [python-barcode](https://github.com/WhyNotHugo/python-barcode)

## Provided Barcodes
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

<!---MIT © Sebastian Beetschen--->