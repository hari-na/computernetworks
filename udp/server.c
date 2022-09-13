#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
	
#define PORT	8080
#define MAXLINE 1024
#define BUF 	256
#define TOT 	20

int main() {
    // converting txt to arr
    char line[TOT][BUF];
    FILE *plist = NULL; 
    int i = 0;
    int total = 0;
    plist = fopen("DNSlist.txt", "r");
    while(fgets(line[i], BUF, plist)) {
        line[i][strlen(line[i]) - 1] = '\0';
        i++;
    }
    total = i;
	int sockfd;
	char buffer[MAXLINE];
	char hello[20];
	struct sockaddr_in servaddr, cliaddr;
		
	if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) {
		perror("Socket not created.");
		exit(EXIT_FAILURE);
	}
		
	memset(&servaddr, 0, sizeof(servaddr));
	memset(&cliaddr, 0, sizeof(cliaddr));
		
	servaddr.sin_family = AF_INET;
	servaddr.sin_addr.s_addr = INADDR_ANY;
	servaddr.sin_port = htons(PORT);
		
	if(bind(sockfd,(const struct sockaddr *)&servaddr, sizeof(servaddr)) < 0){
		perror("bind failed");
		exit(EXIT_FAILURE);
	}
		
	int len, n;
	
	len = sizeof(cliaddr);
	
	n = recvfrom(sockfdrecvfrom, (char *)buffer, MAXLINE,
				MSG_WAITALL, ( struct sockaddr *) &cliaddr,
				&len);
	buffer[n] = '\0';
	printf("Client : %s\n", buffer);
    int chVal = 0;
    for(i = 0; i < total; ++i){
        if(strcmp(line[i], buffer) == 0){
            printf("Your corresponding IP Address for `%s` is: %s\n", line[i], line[i + 1]);
            strcpy(hello, line[i + 1]);
            chVal = 1;
        }
    }
    if(chVal == 0){
        strcpy(hello, "No such website.");
        printf("No such website exists.")
    }
        
	sendto(sockfd, (const char *)hello, strlen(hello), MSG_CONFIRM, (const struct sockaddr *) &cliaddr, len);
		
	return 0;
}
