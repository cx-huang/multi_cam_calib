#include "CCalibBall.h"
#include "AutoThres.cpp"
#include "DetectCircle.cpp"
#include "CCalibBall.cpp"

int main()
{
	char config_file[100] = "config.yml";

	CCalibBall calib(config_file);
	calib.run(config_file);
	return 0;
}