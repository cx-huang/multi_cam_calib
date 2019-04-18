#include "CalibBall.h"

int main()
{
	CalibBall calib_ball("config.yml");
	calib_ball.Run("config.yml");
	return 0;
}