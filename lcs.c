#include <stdio.h>

//gcc -std=c99 -o lcs.so -shared -fPIC lcs.c

//最长公共子串

#define maxn 50000
//int dp[maxn][maxn];
int dp[maxn];

int lcs(int ta[],int tb[],int na,int nb){
	if(na>maxn||nb>maxn){
		printf("[Error]length out of maxn:%d %d\n",na,nb);
		return 0;
	}
	int ans=0;
	for(int j=0;j<nb;++j){
		dp[j] = (ta[0]==tb[j]);
		if(ans<dp[j]) ans = dp[j];
	}
	for(int i=1;i<na;++i){
		for(int j=nb-1;j>0;--j){
			if(ta[i]==tb[j]){
				dp[j] = dp[j-1] + 1;
				if(ans<dp[j]) ans = dp[j];
			}
			else
				dp[j] = 0;
		}
		dp[0] = (ta[i]==tb[0]);
		if(ans<dp[0]) ans = dp[0];
	}
	/*for(int i=0;i<na;++i){
		dp[i][0] = (ta[i]==tb[0]);
		if(ans<dp[i][0]) ans=dp[i][0];
	}
	for(int j=0;j<nb;++j){
		dp[0][j] = (ta[0]==tb[j]);
		if(ans<dp[0][j]) ans=dp[0][j];
	}
	for(int i=1;i<na;++i)
		for(int j=1;j<nb;++j){
			if(ta[i]==tb[j]){
				dp[i][j] = dp[i-1][j-1] + 1;
				if(ans<dp[i][j]) ans=dp[i][j];
			}
			else dp[i][j] = 0;
		}
	*/
	return ans;
}
