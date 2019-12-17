@echo off
SET filename="edf63_ProjectReport"

pdflatex %filename%.tex -aux-directory=build
cd build
bibtex %filename%
cd ..
pdflatex %filename%.tex -aux-directory=build
pdflatex %filename%.tex -aux-directory=build
"%filename%.pdf"
