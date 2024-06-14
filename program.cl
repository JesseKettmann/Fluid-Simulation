__kernel void HelloWorld(__global char* data, __global int* number)
{
data[0] = 'H';
data[1] = 'e';
data[2] = 'l';
data[3] = 'l';
data[4] = 'o';
data[5] = 'W';
data[6] = 'o';
data[7] = 'r';
data[8] = 'l';
data[9] = 'd';
data[10] = '\n';

int i = get_global_id(0);
int num = number[i];
number[i] = num + 3;

}