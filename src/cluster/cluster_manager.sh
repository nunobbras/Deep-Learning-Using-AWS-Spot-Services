#!/usr/bin/env zsh

source ~/.zshrc
cd ~/Dropbox/Research_Code/ML/Nielsen/My_Code
workon research

echo "####################################"
echo "INSTANCES"
echo ""
starcluster listinstances nbbcluster
echo "####################################"

#starcluster addnode -n 3 mycluster
#starcluster stop nbbcluster

function stop_cluster {
	echo "stopping nbbcluster..."
	starcluster stop nbbcluster	
	echo "done"
    return 0
}


function add_nodes {
	echo "stopping nbbcluster..."

	starcluster addnode -n  nbbcluster
	echo "done"
    return 0
}

function login_master {
	echo "login master nbbcluster..."
	starcluster sshmaster nbbcluster -u ubuntu
    return 0
}

