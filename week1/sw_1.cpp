/*
 *2/15 kimhongji
 *deep_study week1
 *sw expert academy no.2072
 */
#include <iostream>
using namespace std;

int main() {

	int num = 0;
	int sum = 0;
	int var = 0;
	cin >> num;
	for (int i = 0; i<num; i++) {
		for (int j = 0; j<10; j++) {
			cin >> var;
			if (var % 2 == 1) sum += var;
		}
		cout << "#" << i + 1 << " " << sum << endl;
		sum = 0;
	}
	return 0;
}
