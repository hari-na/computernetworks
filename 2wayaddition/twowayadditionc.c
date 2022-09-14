// TwoWayAdditionClient

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
#include <arpa/inet.h>
#define MAX 80
#define PORT 8080
#define SA struct sockaddr

void func(int sockfd)
{
    int a, b, answer;

    for (;;)
    {
        printf("Enter the first : ");
        scanf("%d", &a);
        write(sockfd, &a, sizeof(a));

        printf("Enter the second: ");
        scanf("%d", &b);
        write(sockfd, &b, sizeof(b));

        read(sockfd, &answer, sizeof(answer));
        printf("Sum : %d", answer);
    }
}

int main()
{
    int sockfd, connfd;
    struct sockaddr_in servaddr, cli;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1)
    {
        printf("[-]Socket creation failed.\n");
        exit(0);
    }
    else
        printf("[+]Socket successfully created.\n");
    bzero(&servaddr, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    servaddr.sin_port = htons(PORT);

    if (connect(sockfd, (SA *)&servaddr, sizeof(servaddr)) != 0)
    {
        printf("[-]Connection with the server failed.\n");
        exit(0);
    }
    else
        printf("[+]Connected to the server.\n");

    func(sockfd);

    close(sockfd);
}
