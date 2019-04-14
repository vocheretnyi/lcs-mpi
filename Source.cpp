/*
	"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 "C:\Users\Viacheslav\Documents\Visual Studio 2017\Projects\lcs-mpi\x64\Release\lcs-mpi.exe"
*/

#include <iostream>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include <vector>

const int MASTER = 0;
const int N = 4e4;

using namespace std;

int solveSequential(char s1[], char s2[]) {
	vector<vector<int>> dp(3, vector<int>(N + 1, 0));
	for (int sum = 2; sum <= N + N; ++sum) {
		int cur_sum = sum % 3;
		int prev_sum = (cur_sum + 2) % 3;
		int prev_prev_sum = (cur_sum + 1) % 3;
		int start = 1, finish = sum - 1;
		if (sum > N) {
			start = sum - N;
			finish = N;
		}
		for (int i = start; i <= finish; ++i) {
			int j = sum - i;
			if (s1[i - 1] == s2[j - 1]) {
				dp[cur_sum][i] = dp[prev_prev_sum][i - 1] + 1;
			}
			else {
				dp[cur_sum][i] = max(dp[prev_sum][i - 1], dp[prev_sum][i]);
			}
		}
	}
	return dp[(N + N) % 3][N];
}

int main(int argc, char* argv[]) {

	int size, rank;
	int ans;
	double start_time, finish_time, duration;
	char s1[N], s2[N];

	srand(time(NULL));
	cout.precision(4);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;

	int dp[3][N + 1] = {};

	int block_len = N / size;
	if (N % size) {
		block_len++;
	}

	if (rank == MASTER) {
		for (int i = 0; i < N; ++i) {
			char ch;
			ch = static_cast<char>(rand() % 26) + 'a';
			s1[i] = ch;
			ch = static_cast<char>(rand() % 26) + 'a';
			s2[i] = ch;
		}
		start_time = MPI_Wtime();
	}

	MPI_Bcast(&s1[0], N, MPI_CHAR, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(&s2[0], N, MPI_CHAR, MASTER, MPI_COMM_WORLD);

	for (int sum = 2; sum <= N + N; ++sum) {
		int cur_sum = sum % 3;
		int prev_sum = (cur_sum + 2) % 3;
		int prev_prev_sum = (cur_sum + 1) % 3;
		int start = 1, finish = sum - 1;
		if (sum > N) {
			start = sum - N;
			finish = N;
		}

		int l = block_len * rank + 1;
		int r = min(l + block_len - 1, N);

		int L = max(l, start);
		int R = min(r, finish);

		for (int i = L; i <= R; ++i) {
			int j = sum - i;
			if (s1[i - 1] == s2[j - 1]) {
				dp[cur_sum][i] = dp[prev_prev_sum][i - 1] + 1;
			}
			else {
				dp[cur_sum][i] = max(dp[prev_sum][i - 1], dp[prev_sum][i]);
			}
		}

		int dest = rank == size - 1 ? MPI_PROC_NULL : rank + 1;
		MPI_Send(&dp[cur_sum][r], 1, MPI_INT, dest, 1, MPI_COMM_WORLD);

		int src = rank == 0 ? MPI_PROC_NULL : rank - 1;
		MPI_Recv(&dp[cur_sum][l - 1], 1, MPI_INT, src, 1, MPI_COMM_WORLD, &status);
	}

	if (rank == size - 1 && size != 1) {
		MPI_Send(&dp[(N + N) % 3][N], 1, MPI_INT, MASTER, 1, MPI_COMM_WORLD);
	}
	
	if (rank == MASTER) {
		if (size != 1) {
			MPI_Recv(&ans, 1, MPI_INT, size - 1, 1, MPI_COMM_WORLD, &status);
		}
		else {
			ans = dp[(N + N) % 3][N];
		}
		finish_time = MPI_Wtime();
		duration = finish_time - start_time;
		cout << "Parallel  , ans = " << ans << ", time = " << fixed << duration << endl;
		
		start_time = MPI_Wtime();
		ans = solveSequential(s1, s2);
		finish_time = MPI_Wtime();
		duration = finish_time - start_time;
		cout << "Sequential, ans = " << ans << ", time = " << fixed << duration << endl;
	}

	MPI_Finalize();

	return 0;
}
