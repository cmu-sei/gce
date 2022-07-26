@REM Generalized Calibration Error

@REM Copyright 2022 Carnegie Mellon University.

@REM NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE
@REM MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO
@REM WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER 
@REM INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR 
@REM MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. 
@REM CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT
@REM TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

@REM Released under a MIT (SEI)-style license, please see license.txt or contact 
@REM permission@sei.cmu.edu for full terms.

@REM [DISTRIBUTION STATEMENT A] This material has been approved for public release 
@REM and unlimited distribution.  Please see Copyright notice for non-US Government 
@REM use and distribution.

@REM This Software includes and/or makes use of the following Third-Party Software 
@REM subject to its own license:

@REM 1. calibration (https://github.com/uu-sml/calibration/blob/master/LICENSE) 
@REM Copyright 2019 Carl Andersson, David Widmann.

@REM 2. NumPy (https://github.com/numpy/numpy/blob/main/LICENSE.txt) 
@REM Copyright 2005-2022 NumPy Developers.

@REM DM22-0406


@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
